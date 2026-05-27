import asyncio
import os
import re
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

import numpy as np
from ray import serve
from ray.serve import handle

from src.core.module import Module, ModuleWithHandle


# Defaults — overridden by env vars in production (see README.md)
_MODEL_PATH = os.environ.get("HURI_MODEL_PATH", "/models/cosytts/iic/CosyVoice2-0.5B")
_VOICE_SAMPLE_PATH = os.environ.get("HURI_VOICE_SAMPLE_PATH", "/assets/voice.wav")
_VOICE_SAMPLE_TRANSCRIPT = os.environ.get(
    "HURI_VOICE_TRANSCRIPT", "Hello, this is my voice sample for cloning."
)

# Hard endings (.!?) trigger synthesis immediately; soft endings (,;:) only after
# min_clause_chars are buffered, to avoid synthesizing very short fragments.
_HARD_END_RE = re.compile(r'[.!?]["\']?\s+')
_SOFT_END_RE = re.compile(r'[,;:]\s+')

_DONE = object()  # sentinel for exhausted sync generator


@dataclass
class Token:
    text: str
    end: bool  # True on the last token of an LLM stream


@dataclass
class Audio:
    data: np.ndarray  # float32, values in [-1.0, 1.0]
    sample_rate: int
    end: bool = False  # True on the last chunk of an utterance


@serve.deployment(name="TTS")
class TTSDeployment:
    def __init__(
        self,
        model_path: str = _MODEL_PATH,
        voice_sample_path: str = _VOICE_SAMPLE_PATH,
        voice_sample_transcript: str = _VOICE_SAMPLE_TRANSCRIPT,
    ):
        from cosyvoice.cli.cosyvoice import CosyVoice2
        from cosyvoice.utils.file_utils import load_wav

        self.model = CosyVoice2(model_path, load_jit=False, load_trt=False)
        self.sample_rate: int = self.model.sample_rate

        self.prompt_speech = load_wav(voice_sample_path, 16000)
        self.prompt_text: str = voice_sample_transcript

    async def synthesize(self, text: str) -> AsyncGenerator[Audio, None]:
        """Run CosyVoice2 streaming inference and yield Audio chunks.

        The synchronous CosyVoice2 generator runs in a thread-pool executor so
        it does not block the asyncio event loop between chunks.
        """
        loop = asyncio.get_running_loop()
        gen = self.model.inference_zero_shot(
            text,
            self.prompt_text,
            self.prompt_speech,
            stream=True,
        )
        while True:
            result = await loop.run_in_executor(None, next, gen, _DONE)
            if result is _DONE:
                break
            yield Audio(
                data=result["tts_speech"].squeeze(0).numpy().astype(np.float32),
                sample_rate=self.sample_rate,
            )

    async def get_sample_rate(self) -> int:
        return self.sample_rate


class TTS(ModuleWithHandle):
    """TTS Module

    Stream text tokens in, stream audio chunks out using CosyVoice2 zero-shot
    voice cloning.

    Buffers incoming tokens and synthesizes as soon as a sentence or clause
    boundary is detected. Audio chunks are yielded immediately as CosyVoice2
    produces them, so playback can start before synthesis is complete.

    Compatible with both the Ray Serve event graph (async generator support in
    EventGraph._run) and direct client streaming.

    input: token (Token),
    output: audio (Audio)

    :min_clause_chars: minimum buffer length before a soft boundary (,;:)
        triggers synthesis. Hard endings (.!?) always trigger immediately.
        Raise this value to produce longer, more natural-sounding segments.
    """

    _handle_cls = TTSDeployment
    input_type = "token"
    output_type = "audio"

    def __init__(
        self,
        handle: handle.DeploymentHandle,
        min_clause_chars: int = 20,
    ):
        super().__init__(handle)
        self.min_clause_chars: int = min_clause_chars
        self._buffer: str = ""

    async def process(self, token: Token) -> AsyncGenerator[Audio, None]:  # type: ignore[override]
        self._buffer += token.text

        # Drain all complete clauses from the buffer before waiting for more tokens
        while True:
            clause, remainder = self._split(self._buffer)
            if not clause:
                break
            self._buffer = remainder
            async for chunk in self.handle.synthesize.remote(clause):
                yield chunk

        # Flush the remaining buffer when the LLM stream ends
        if token.end and self._buffer.strip():
            async for chunk in self.handle.synthesize.remote(self._buffer.strip()):
                yield chunk
            self._buffer = ""
        if token.end:
            sample_rate = await self.handle.get_sample_rate.remote()
            yield Audio(data=np.array([], dtype=np.float32), sample_rate=sample_rate, end=True)

    def _split(self, text: str) -> tuple[str, str]:
        """Return (clause_to_synthesize, remaining_buffer).

        Splits on the first hard sentence ending (.!?) unconditionally, or on
        the first soft clause ending (,;:) once the buffer is long enough.
        Returns ("", text) when no boundary is found.
        """
        m = _HARD_END_RE.search(text)
        if m:
            return text[: m.end()].strip(), text[m.end() :]

        if len(text) >= self.min_clause_chars:
            m = _SOFT_END_RE.search(text)
            if m:
                return text[: m.end()].strip(), text[m.end() :]

        return "", text
