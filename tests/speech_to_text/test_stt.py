import asyncio

import numpy as np
import pytest

from src.modules.speech_to_text.events import Transcript, Voice
from src.modules.speech_to_text.speech_to_text import STT, STTDeployment

from .conftest import FRAME, FakeHandle, voice_frame

# transcribe_window=2.0 s -> 32000 samples: the 67th 480-sample frame fills it.
WINDOW_FRAMES = 67


def make_stt(handle: FakeHandle, **kwargs) -> STT:
    args = dict(language="en", block_duration=0.03)
    args.update(kwargs)
    return STT(_handle=handle, **args)


async def feed(stt: STT, n: int, value: float = 0.1):
    out = []
    for _ in range(n):
        out.append(await stt.process(Voice(voice_frame(value))))
    return out


async def settle():
    for _ in range(3):
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# the confirmed race
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_end_marker_survives_inflight_partial(fake_handle):
    """Voice(None) then a speech frame arrive while a partial transcription is
    in flight. The old STT lost the end marker here (a shared `silence` flag
    was flipped back by the speech frame) and the turn never closed."""
    fake_handle.gate = asyncio.Event()
    stt = make_stt(fake_handle)

    # Fill the window: the last frame starts a partial that blocks on the gate.
    partial_task = asyncio.create_task(_feed_and_return_last(stt, WINDOW_FRAMES))
    await settle()
    assert [c.final for c in fake_handle.calls] == [False]
    assert stt.running

    # End of utterance while the partial is pending...
    end_task = asyncio.create_task(stt.process(Voice(None)))
    await settle()
    assert [c.final for c in fake_handle.calls] == [False, True]
    assert stt.turn_id == 2

    # ...and the user starts the next utterance before anything completes.
    assert await stt.process(Voice(voice_frame(0.2))) is None
    assert len(stt.utterance) == 1

    fake_handle.gate.set()
    partial_result = await partial_task
    end_result = await end_task

    assert partial_result is None  # stale partial dropped
    assert end_result == Transcript("FINAL", True)
    assert not stt.running
    # nothing of turn 0 leaked into turn 1
    assert stt.partials == []
    assert len(stt.utterance) == 1
    assert stt.utterance[0][0] == pytest.approx(0.2)
    # the final transcribed the whole turn 0
    assert len(fake_handle.final_calls[0].audio) == WINDOW_FRAMES * FRAME


async def _feed_and_return_last(stt: STT, n: int):
    return (await feed(stt, n))[-1]


# ---------------------------------------------------------------------------
# turn contents
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_final_uses_all_frames_including_after_window(fake_handle):
    stt = make_stt(fake_handle)
    out = await feed(stt, 70)
    assert out[WINDOW_FRAMES - 1] == Transcript("partial", False)
    assert sum(o is not None for o in out) == 1

    result = await stt.process(Voice(None))
    assert result == Transcript("FINAL", True)
    assert len(fake_handle.final_calls) == 1
    assert len(fake_handle.final_calls[0].audio) == 70 * FRAME


@pytest.mark.asyncio
async def test_frames_during_final_belong_to_next_turn(fake_handle):
    fake_handle.gate = asyncio.Event()
    stt = make_stt(fake_handle)
    await feed(stt, 10, value=0.1)
    end_task = asyncio.create_task(stt.process(Voice(None)))
    await settle()
    assert len(fake_handle.final_calls) == 1

    await feed(stt, 5, value=0.5)
    fake_handle.gate.set()
    assert await end_task == Transcript("FINAL", True)
    assert len(fake_handle.final_calls[0].audio) == 10 * FRAME
    assert np.all(fake_handle.final_calls[0].audio == pytest.approx(0.1))

    result = await stt.process(Voice(None))
    assert result == Transcript("FINAL", True)
    assert len(fake_handle.final_calls[1].audio) == 5 * FRAME
    assert np.all(fake_handle.final_calls[1].audio == pytest.approx(0.5))


@pytest.mark.asyncio
async def test_no_partial_while_final_pending(fake_handle):
    fake_handle.gate = asyncio.Event()
    stt = make_stt(fake_handle)
    await feed(stt, 10)
    end_task = asyncio.create_task(stt.process(Voice(None)))
    await settle()
    assert [c.final for c in fake_handle.calls] == [True]

    await feed(stt, 70)  # a full window, but the final holds the slot
    assert [c.final for c in fake_handle.calls] == [True]

    fake_handle.gate.set()
    await end_task
    await feed(stt, 1)
    assert [c.final for c in fake_handle.calls] == [True, False]


@pytest.mark.asyncio
async def test_two_short_turns_finals_are_fifo():
    handle = FakeHandle(final_text=lambda call: f"len{len(call.audio)}")
    handle.gate = asyncio.Event()
    stt = make_stt(handle)

    await feed(stt, 5)
    end1 = asyncio.create_task(stt.process(Voice(None)))
    await settle()
    await feed(stt, 3)
    end2 = asyncio.create_task(stt.process(Voice(None)))
    await settle()
    # the second final waits for the first: one call in flight
    assert len(handle.final_calls) == 1

    handle.gate.set()
    r1, r2 = await asyncio.gather(end1, end2)
    assert r1 == Transcript(f"len{5 * FRAME}", True)
    assert r2 == Transcript(f"len{3 * FRAME}", True)
    assert [len(c.audio) for c in handle.final_calls] == [5 * FRAME, 3 * FRAME]


@pytest.mark.asyncio
async def test_partial_completing_after_close_is_not_appended_to_new_turn(fake_handle):
    fake_handle.gate = asyncio.Event()
    stt = make_stt(fake_handle)
    partial_task = asyncio.create_task(_feed_and_return_last(stt, WINDOW_FRAMES))
    await settle()
    fake_handle.gate.set()
    end_result = await stt.process(Voice(None))  # closes turn 0 first
    assert end_result == Transcript("FINAL", True)
    assert await partial_task is None
    assert stt.partials == []
    assert stt.turn_id == 2


# ---------------------------------------------------------------------------
# failure paths never wedge the module
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_partial_exception_clears_running(fake_handle):
    fake_handle.error = RuntimeError("libcublas.so.12 not found")
    stt = make_stt(fake_handle)
    out = await feed(stt, WINDOW_FRAMES)
    assert out[-1] is None
    assert not stt.running

    fake_handle.error = None
    # the window slid nothing (call failed), so the next frame retries
    out = await feed(stt, 1)
    assert out[-1] == Transcript("partial", False)
    assert len(fake_handle.partial_calls) == 2


@pytest.mark.asyncio
async def test_partial_timeout_cancels_and_clears_running(fake_handle):
    fake_handle.gate = asyncio.Event()  # never released
    stt = make_stt(fake_handle, partial_timeout=0.05)
    out = await feed(stt, WINDOW_FRAMES)
    assert out[-1] is None
    assert not stt.running
    assert fake_handle.cancelled == 1


@pytest.mark.asyncio
async def test_final_timeout_falls_back_to_partials():
    handle = FakeHandle(partial_text=lambda call: f"p{len(handle.partial_calls)}")
    stt = make_stt(handle, final_timeout=0.05)
    await feed(stt, WINDOW_FRAMES)  # -> "p1"
    await feed(stt, 34)  # slide of 1 s = 33.3 frames -> second partial "p2"
    assert stt.partials == ["p1", "p2"]

    handle.gate = asyncio.Event()  # the final hangs
    result = await stt.process(Voice(None))
    assert result == Transcript("p1 p2", True)
    assert handle.cancelled == 1
    assert not stt._final_lock.locked()


@pytest.mark.asyncio
async def test_final_exception_falls_back_to_partials_and_keeps_end(fake_handle):
    stt = make_stt(fake_handle)
    await feed(stt, 5)  # no partial: nothing to fall back to
    fake_handle.error = RuntimeError("replica died")
    result = await stt.process(Voice(None))
    assert result == Transcript("", True)  # end=True even when empty
    assert not stt._final_lock.locked()


@pytest.mark.asyncio
async def test_empty_turn_end_marker_is_ignored(fake_handle):
    stt = make_stt(fake_handle)
    assert await stt.process(Voice(None)) is None
    assert fake_handle.calls == []
    assert stt.turn_id == 1


# ---------------------------------------------------------------------------
# buffering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_partial_backlog_skips_ahead(fake_handle):
    fake_handle.gate = asyncio.Event()
    stt = make_stt(fake_handle)
    partial_task = asyncio.create_task(_feed_and_return_last(stt, WINDOW_FRAMES))
    await settle()
    # inference is stuck; 3 more windows of audio arrive meanwhile
    await feed(stt, 3 * WINDOW_FRAMES, value=0.3)
    fake_handle.gate.set()
    await partial_task
    # next frame triggers a partial over the freshest window only
    await feed(stt, 1, value=0.3)
    assert len(fake_handle.partial_calls) == 2
    audio = fake_handle.partial_calls[1].audio
    assert len(audio) == stt.window_samples
    assert np.all(audio == pytest.approx(0.3))
    assert len(stt.window_buf) <= stt.window_samples + stt.slide_samples


@pytest.mark.asyncio
async def test_utterance_capped_keeps_tail(fake_handle):
    stt = make_stt(fake_handle, max_final_duration=0.5, enable_partials=False)
    cap = stt.max_final_samples  # 8000 samples
    for i in range(40):  # 19200 samples
        await stt.process(Voice(voice_frame(float(i))))
    await stt.process(Voice(None))
    audio = fake_handle.final_calls[0].audio
    assert len(audio) <= cap
    assert audio[-1] == 39.0
    assert audio[0] >= 39 - cap / FRAME


@pytest.mark.asyncio
async def test_enable_partials_false_never_calls_partial(fake_handle):
    stt = make_stt(fake_handle, enable_partials=False)
    out = await feed(stt, 3 * WINDOW_FRAMES)
    assert all(o is None for o in out)
    assert fake_handle.calls == []
    assert await stt.process(Voice(None)) == Transcript("FINAL", True)


# ---------------------------------------------------------------------------
# deployment flags
# ---------------------------------------------------------------------------


class _FakeSegment:
    def __init__(self, text):
        self.text = text


class _FakeWhisper:
    def __init__(self):
        self.kwargs = []

    def transcribe(self, audio, **kwargs):
        self.kwargs.append(kwargs)
        return iter([_FakeSegment(" hello "), _FakeSegment("world ")]), None


def test_deployment_final_and_partial_kwargs():
    cls = STTDeployment.func_or_class
    deployment = cls.__new__(cls)
    deployment.model_faster = _FakeWhisper()
    audio = np.zeros(16000, dtype=np.float32)

    assert deployment._transcribe_sync(audio, "en", final=True) == "hello world"
    assert deployment._transcribe_sync(audio, "en", final=False) == "hello world"

    final_kwargs, partial_kwargs = deployment.model_faster.kwargs
    assert final_kwargs["beam_size"] == 5
    assert final_kwargs["vad_filter"] is True
    assert final_kwargs["condition_on_previous_text"] is False
    assert partial_kwargs["beam_size"] == 1
    assert "vad_filter" not in partial_kwargs
    assert partial_kwargs["language"] == "en"
