import asyncio
from typing import List, Optional

import numpy as np
from faster_whisper import WhisperModel

from src.core.module import Module

from .events import Transcript, Voice


class STT(Module):
    """STT Module

    Transcribe voice using Faster_Whisper.

    input: voice,
    output: transcript

    :model: size of the model to use (tiny, tiny.en, base, base.en, small,
        small.en, distil-small.en, medium, medium.en, distil-medium.en,
        large-v1, large-v2, large-v3, large, distil-large-v2, distil-large-v3,
        large-v3-turbo, or turbo).
    :language: language spoken in the audio. It should be a language code such
        as "en" or "fr".
    :sample_rate: size of received voice audio. Usually 8000, 16000 or 48000.
    :block_duration: size of received voice audio (in s).
    """

    input_type = "voice"
    output_type = "transcript"

    def __init__(
        self,
        model: str = "base",
        language: str = "en",
        sample_rate: int = 16000,
        block_duration: float = 0.020,  # s
        transcribe_window: float = 2.0,  # s
        transcribe_step: float = 1.0,  # s
    ):
        super().__init__()

        self.model_faster = WhisperModel(model, cpu_threads=2)
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

        self.pending_silence = False
        processing_audio = np.concatenate(processing_chunks, axis=0)

        def transcribe_text():
            segments, _ = self.model_faster.transcribe(
                processing_audio,
                language=self.language,
                beam_size=1,
            )
            return " ".join(seg.text for seg in segments).strip()

        current_text = await asyncio.to_thread(transcribe_text)

        processed_size = self.window_size - self.step_size
        async with self.lock:
            self.buffer = self.buffer[processed_size:]
            self.running = False

        return Transcript(current_text, self.silence)
