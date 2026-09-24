"""Shared fakes for the MIC / STT / TAG tests.

Nothing here touches webrtcvad's decision or the Whisper model: MIC gets a
FakeVad whose verdict is "any non-zero sample", STT gets a FakeHandle whose
`transcribe.remote()` can be gated on an asyncio.Event so a test can inject
events while a transcription is "in flight" — which is exactly the situation
the old STT lost its end-of-turn marker in.
"""

import asyncio
from typing import Callable, List, Optional, Union

import numpy as np
import pytest

FRAME = 480  # 30 ms @ 16 kHz, what the browser client sends


def speech_frame(value: int = 1000) -> bytes:
    return np.full(FRAME, value, dtype=np.int16).tobytes()


def silence_frame() -> bytes:
    return bytes(FRAME * 2)


def voice_frame(value: float = 0.1, n: int = FRAME) -> np.ndarray:
    return np.full(n, value, dtype=np.float32)


class FakeVad:
    """webrtcvad stand-in: speech iff any sample is non-zero; rejects frame
    lengths webrtcvad would reject."""

    def is_speech(self, raw: bytes, sample_rate: int) -> bool:
        valid = {2 * sample_rate * ms // 1000 for ms in (10, 20, 30)}
        if len(raw) not in valid:
            raise Exception("Error while processing frame")
        return bool(np.any(np.frombuffer(raw, dtype=np.int16)))


class FakeCall:
    def __init__(self, audio: np.ndarray, language: str, final: bool):
        self.audio = audio
        self.language = language
        self.final = final


class FakeResponse:
    """Awaitable like ray.serve's DeploymentResponse, with .cancel()."""

    def __init__(self, handle: "FakeHandle", call: FakeCall):
        self._handle = handle
        self._call = call
        self.cancelled = False

    def __await__(self):
        return self._run().__await__()

    async def _run(self) -> str:
        if self._handle.gate is not None:
            await self._handle.gate.wait()
        if self._handle.error is not None:
            raise self._handle.error
        text = self._handle.final_text if self._call.final else self._handle.partial_text
        return text(self._call) if callable(text) else text

    def cancel(self) -> None:
        self.cancelled = True
        self._handle.cancelled += 1


class _FakeTranscribe:
    def __init__(self, handle: "FakeHandle"):
        self._handle = handle

    def remote(self, audio, language, final: bool = False) -> FakeResponse:
        call = FakeCall(np.asarray(audio), language, final)
        self._handle.calls.append(call)
        response = FakeResponse(self._handle, call)
        self._handle.responses.append(response)
        return response


TextSpec = Union[str, Callable[[FakeCall], str]]


class FakeHandle:
    def __init__(self, partial_text: TextSpec = "partial", final_text: TextSpec = "FINAL"):
        self.calls: List[FakeCall] = []
        self.responses: List[FakeResponse] = []
        self.gate: Optional[asyncio.Event] = None
        self.error: Optional[BaseException] = None
        self.cancelled = 0
        self.partial_text = partial_text
        self.final_text = final_text
        self.transcribe = _FakeTranscribe(self)

    @property
    def partial_calls(self) -> List[FakeCall]:
        return [c for c in self.calls if not c.final]

    @property
    def final_calls(self) -> List[FakeCall]:
        return [c for c in self.calls if c.final]


@pytest.fixture
def fake_handle() -> FakeHandle:
    return FakeHandle()
