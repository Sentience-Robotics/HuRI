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
    """faster-whisper model wrapper.

    Holds the WhisperModel and runs transcription on its own Ray Serve actor,
    off the HuRI master actor — model load and GPU inference no longer block the
    websocket ingress / per-session router. Pinned to a GPU worker via
    ray_actor_options in the Serve config (see deploy values.yaml).

    Stateless across calls: the per-session sliding-window buffering lives in the
    STT module, so this deployment is shared across all sessions.

    :model: path to (or size name of) the faster-whisper model. Defaults to the
        HURI_STT_MODEL_PATH env var, falling back to "base".
    :device: "cpu", "cuda", or "auto".
    :compute_type: e.g. "int8", "float16", or "auto".
    """

    def __init__(
        self,
        model: str = _MODEL_PATH,
        device: str = "auto",
        compute_type: str = "auto",
    ):
        from faster_whisper import WhisperModel

        self.model_faster = WhisperModel(
            model,
            device=device,
            compute_type=compute_type,
        )

    async def transcribe(self, audio: np.ndarray, language: str = "en") -> str:
        segments, _ = self.model_faster.transcribe(
            audio,
            language=language,
            beam_size=1,  # faster for realtime
        )
        return " ".join([seg.text for seg in segments]).strip()


class STT(ModuleWithHandle):
    """STT Module

    Transcribe voice using Faster_Whisper.

    Holds the per-session sliding-window buffer and delegates the actual
    transcription to a handle-backed STTDeployment, so the Whisper model runs
    off the HuRI master node.

    input: voice,
    output: transcript

    :language: language spoken in the audio. It should be a language code such
        as "en" or "fr".
    :sample_rate: size of received voice audio. Usually 8000, 16000 or 48000.
    :block_duration: size of received voice audio (in s).
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
        **kwargs,
    ):
        super().__init__(_handle=_handle, **kwargs)

        self.language = language

        self.sample_rate = sample_rate
        self.window_size: int = int(transcribe_window / block_duration)
        self.step_size: int = int(transcribe_step / block_duration)

        self.buffer: List[np.ndarray] = []

        self.silence: bool = True

        self.prev_text: str = ""
        self.stable_text: str = ""

        self.running = False
        self.lock: asyncio.Lock = asyncio.Lock()

    async def process(self, voice: Voice) -> Optional[Transcript]:  # type: ignore[override]
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

        self.pending_silence = False
        processing_audio = np.concatenate(processing_chunks, axis=0)

        current_text = await self._handle.transcribe.remote(
            processing_audio, self.language
        )

        processed_size = self.window_size - self.step_size
        async with self.lock:
            self.buffer = self.buffer[processed_size:]
            self.running = False

        return Transcript(current_text, self.silence)
