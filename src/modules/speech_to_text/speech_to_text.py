import asyncio
import os
from typing import List, Optional

import numpy as np
from ray import serve
from ray.serve import handle

from src.core.module import ModuleWithHandle

from .events import Transcript, Voice

_MODEL_PATH = os.environ.get("HURI_STT_MODEL_PATH", "base")


@serve.deployment(name="STT")
class STTDeployment:
    """Stateless Whisper inference actor.

    Holds the faster-whisper model in a single Ray actor (pinned to the AMD
    worker in deployment configs). Exposes a single transcribe() call so
    per-session STT clients can offload the heavy work without owning a GPU.

    HURI_STT_MODEL_PATH: path to a local faster-whisper model directory (from
    the whisper PVC). Falls back to "base" which triggers a HuggingFace
    download — only acceptable for local dev without a PVC.
    """

    def __init__(
        self,
        model: str = _MODEL_PATH,
        device: str = "auto",
        compute_type: str = "auto",
    ):
        print(f"[STT] loading model from {model!r} (device={device} compute_type={compute_type})", flush=True)
        from faster_whisper import WhisperModel

        self.model_faster = WhisperModel(
            model,
            device=device,
            compute_type=compute_type,
        )
        print(f"[STT] model loaded", flush=True)
        self.language = "en"

    async def transcribe(self, audio: np.ndarray) -> str:
        loop = asyncio.get_running_loop()
        segments, _ = await loop.run_in_executor(
            None,
            lambda: self.model_faster.transcribe(
                audio,
                language=self.language,
                beam_size=1,
            ),
        )
        return " ".join(seg.text for seg in segments).strip()


class STT(ModuleWithHandle):
    """STT Module

    Per-session client: keeps the rolling window / silence state, offloads each
    transcription window to the shared STTDeployment actor.

    input: voice,
    output: transcript

    :sample_rate: size of received voice audio. Usually 8000, 16000 or 48000.
    :block_duration: size of received voice audio (in s).
    :transcribe_window: rolling window length (s) handed to Whisper.
    :transcribe_step: stride (s) between successive windows.
    """

    _handle_cls = STTDeployment
    input_type = "voice"
    output_type = "transcript"

    def __init__(
        self,
        _handle: handle.DeploymentHandle,
        language: str = "en",
        sample_rate: int = 16000,
        block_duration: float = 0.020,  # s
        transcribe_window: float = 2.0,  # s
        transcribe_step: float = 1.0,  # s
    ):
        super().__init__(_handle=_handle)

        self.language = language
        self.sample_rate = sample_rate
        self.window_size: int = int(transcribe_window / block_duration)
        self.step_size: int = int(transcribe_step / block_duration)

        self.buffer: List[np.ndarray] = []

        self.silence: bool = True

        self.running = False
        self.lock: asyncio.Lock = asyncio.Lock()

    async def process(self, voice: Voice) -> Optional[Transcript]:
        if voice.data is None:
            self.silence = True
        else:
            self.silence = False
            async with self.lock:
                self.buffer.append(voice.data)

        async with self.lock:
            if self.running:
                return None
            self.running = True

        async with self.lock:
            buffer_size = len(self.buffer)
            if buffer_size == 0 or (
                self.silence is False and buffer_size < self.window_size
            ):
                self.running = False
                return None
            processing_chunks = self.buffer[: self.window_size]

        processing_audio = np.concatenate(processing_chunks, axis=0)

        current_text: str = await self._handle.transcribe.remote(processing_audio)

        processed_size = self.window_size - self.step_size
        async with self.lock:
            self.buffer = self.buffer[processed_size:]
            self.running = False

        return Transcript(current_text, self.silence)
