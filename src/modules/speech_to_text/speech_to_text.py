import asyncio
import os
from typing import AsyncGenerator, List

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

        # Set when the VAD emits its single end-of-utterance marker (Voice(None)).
        # Remembered rather than acted on immediately so it survives an in-flight
        # transcribe — otherwise the terminal window (and thus the question that
        # drives the RAG) is silently dropped.
        self.pending_end: bool = False

        self.running = False
        self.lock: asyncio.Lock = asyncio.Lock()

    async def process(self, voice: Voice) -> AsyncGenerator[Transcript, None]:  # type: ignore[override]
        async with self.lock:
            if voice.data is None:
                self.pending_end = True
            else:
                self.buffer.append(voice.data)

            if self.running:
                # Another invocation owns the drain loop below; it will pick up
                # the frame we just buffered (and any pending end-of-utterance).
                return
            self.running = True

        try:
            while True:
                async with self.lock:
                    end = self.pending_end
                    buffer_size = len(self.buffer)

                    if buffer_size == 0:
                        # Nothing left to transcribe. If the utterance just
                        # ended, still emit a terminal transcript so the
                        # aggregator finalises the question.
                        if end:
                            self.pending_end = False
                            yield Transcript("", True)
                        return

                    # Mid-speech: hold until a full window has accumulated.
                    if not end and buffer_size < self.window_size:
                        return

                    processing_chunks = self.buffer[: self.window_size]
                    # On end-of-utterance, the last window drains the buffer.
                    final = end and buffer_size <= self.window_size

                processing_audio = np.concatenate(processing_chunks, axis=0)
                current_text: str = await self._handle.transcribe.remote(
                    processing_audio
                )

                async with self.lock:
                    if final:
                        self.buffer = []
                        self.pending_end = False
                    else:
                        self.buffer = self.buffer[self.window_size - self.step_size :]

                yield Transcript(current_text, final)
                if final:
                    return
        finally:
            async with self.lock:
                self.running = False
