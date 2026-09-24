import json
import wave

import pytest

from src.core.events import BytesEvent
from src.modules.speech_to_text.microphone_vad import MIC

from .conftest import FRAME, FakeVad, silence_frame, speech_frame


def make_mic(**kwargs) -> MIC:
    args = dict(vad_agressiveness=2, silence_duration=1.0, block_duration=0.030)
    args.update(kwargs)
    mic = MIC(**args)
    mic.vad = FakeVad()
    return mic


async def feed(mic: MIC, frames):
    return [await mic.process(BytesEvent(data=f)) for f in frames]


async def open_turn(mic: MIC):
    """Two speech frames: the debounce threshold."""
    out = await feed(mic, [speech_frame(), speech_frame()])
    assert out[0] is None
    assert out[1] is not None and out[1].data is not None
    assert mic.in_utterance
    return out[1]


# silence_duration=1.0 s at 480-sample frames: the 34th trailing silence frame
# is the first one at or past 16000 samples.
CLOSE_AFTER = 34


@pytest.mark.asyncio
async def test_onset_needs_debounce_and_emits_preroll():
    mic = make_mic()
    out = await feed(mic, [silence_frame()] * 5 + [speech_frame()])
    assert all(v is None for v in out)
    assert not mic.in_utterance

    onset = await mic.process(BytesEvent(data=speech_frame()))
    assert onset is not None and onset.data is not None
    # pre-roll deque holds 10 frames; only 7 were seen
    assert len(onset.data) == 7 * FRAME
    assert onset.data.dtype.name == "float32"
    assert mic.in_utterance
    assert mic.turn == 1


@pytest.mark.asyncio
async def test_isolated_blip_outside_utterance_emits_nothing():
    mic = make_mic()
    out = await feed(mic, [speech_frame()] + [silence_frame()] * 5)
    assert all(v is None for v in out)
    assert not mic.in_utterance


@pytest.mark.asyncio
async def test_non_speech_frames_inside_utterance_are_forwarded():
    mic = make_mic()
    await open_turn(mic)
    voice = await mic.process(BytesEvent(data=silence_frame()))
    assert voice is not None and voice.data is not None
    assert len(voice.data) == FRAME


@pytest.mark.asyncio
async def test_close_after_silence_duration_counts_real_frames_once():
    mic = make_mic()
    await open_turn(mic)
    out = await feed(mic, [silence_frame()] * CLOSE_AFTER)
    assert all(v is not None and v.data is not None for v in out[:-1])
    assert out[-1] is not None and out[-1].data is None
    assert not mic.in_utterance
    # sent only once: further silence is dropped
    assert await feed(mic, [silence_frame()] * 10) == [None] * 10


@pytest.mark.asyncio
async def test_lone_blip_does_not_reset_silence_countdown():
    mic = make_mic()
    await open_turn(mic)
    await feed(mic, [silence_frame()] * 20)
    blip = await mic.process(BytesEvent(data=speech_frame()))
    assert blip is not None and blip.data is not None  # forwarded, but no reset
    # 20 * 480 = 9600 already counted: 14 more frames reach 16000
    out = await feed(mic, [silence_frame()] * 14)
    assert out[-2] is not None and out[-2].data is not None
    assert out[-1] is not None and out[-1].data is None


@pytest.mark.asyncio
async def test_two_speech_frames_reset_silence_countdown():
    mic = make_mic()
    await open_turn(mic)
    await feed(mic, [silence_frame()] * 20)
    await feed(mic, [speech_frame(), speech_frame()])
    assert mic.silence_samples == 0
    out = await feed(mic, [silence_frame()] * CLOSE_AFTER)
    assert all(v is not None and v.data is not None for v in out[:-1])
    assert out[-1] is not None and out[-1].data is None


@pytest.mark.asyncio
async def test_max_utterance_forces_close_and_reseeds_next_turn():
    mic = make_mic(max_utterance_duration=1.0)  # 16000 samples
    await open_turn(mic)  # 960 samples so far
    closed = None
    n = 0
    for n in range(1, 40):
        v = await mic.process(BytesEvent(data=speech_frame(7)))
        if v is not None and v.data is None:
            closed = v
            break
    assert closed is not None
    assert n == 32  # 960 + 32 * 480 >= 16000
    assert not mic.in_utterance
    # Still talking: the next speech frame opens turn 2, carrying the frame
    # that triggered the close so nothing is lost.
    onset = await mic.process(BytesEvent(data=speech_frame(9)))
    assert onset is not None and onset.data is not None
    assert len(onset.data) == 2 * FRAME
    assert onset.data[0] == pytest.approx(7 / 32768)
    assert onset.data[-1] == pytest.approx(9 / 32768)
    assert mic.turn == 2


@pytest.mark.asyncio
async def test_preroll_cleared_on_close():
    mic = make_mic()
    await open_turn(mic)
    await feed(mic, [silence_frame()] * CLOSE_AFTER)
    assert len(mic.pre_roll) == 0
    onset = (await feed(mic, [speech_frame(), speech_frame()]))[-1]
    assert onset is not None and len(onset.data) == 2 * FRAME


# ---------------------------------------------------------------------------
# flush: an empty frame (the client closed its mic) ends the turn now
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flush_closes_open_turn_without_waiting_for_silence():
    mic = make_mic()
    await open_turn(mic)
    await feed(mic, [speech_frame()] * 10 + [silence_frame()] * 5)  # < 1 s
    marker = await mic.process(BytesEvent(data=b""))
    assert marker is not None and marker.data is None
    assert not mic.in_utterance
    assert mic.turn == 1
    # exactly once: a second flush (or trailing silence) emits nothing more
    assert await mic.process(BytesEvent(data=b"")) is None
    assert await feed(mic, [silence_frame()] * 5) == [None] * 5


@pytest.mark.asyncio
async def test_flush_outside_turn_is_noop_but_forgets_preroll():
    mic = make_mic()
    await feed(mic, [silence_frame()] * 8 + [speech_frame()])  # one blip
    assert len(mic.pre_roll) == 9 and mic.speech_run == 1
    assert await mic.process(BytesEvent(data=b"")) is None
    assert not mic.in_utterance and mic.turn == 0
    assert len(mic.pre_roll) == 0 and mic.speech_run == 0
    # the next recording starts clean: debounce from scratch, no stale pre-roll
    out = await feed(mic, [speech_frame(), speech_frame()])
    assert out[0] is None
    assert out[1] is not None and len(out[1].data) == 2 * FRAME


@pytest.mark.asyncio
async def test_next_recording_after_flush_is_a_fresh_turn():
    mic = make_mic()
    await open_turn(mic)
    await feed(mic, [speech_frame()] * 10)
    await mic.process(BytesEvent(data=b""))
    onset = (await feed(mic, [speech_frame(), speech_frame()]))[-1]
    assert onset is not None and onset.data is not None
    assert len(onset.data) == 2 * FRAME
    assert mic.turn == 2


@pytest.mark.asyncio
async def test_flush_is_not_an_undecodable_frame(tmp_path):
    mic = make_mic(dump_dir=str(tmp_path))
    await open_turn(mic)
    await mic.process(BytesEvent(data=b""))
    assert not mic._bad_frame_logged
    await mic.finalize()
    # the marker carries no audio: nothing lands in the dump
    wav = next(tmp_path.glob("mic_*.wav"))
    with wave.open(str(wav), "rb") as w:
        assert w.getnframes() == 2 * FRAME
    assert len(open(str(wav) + ".jsonl").readlines()) == 2


@pytest.mark.asyncio
async def test_rms_gate_treats_quiet_vad_positive_frames_as_silence():
    mic = make_mic(min_rms_dbfs=-30.0)
    # 100/32768 ~ -50 dBFS: below the floor
    assert await feed(mic, [speech_frame(100)] * 3) == [None] * 3
    assert not mic.in_utterance
    # 3000/32768 ~ -21 dBFS: above it
    out = await feed(mic, [speech_frame(3000)] * 2)
    assert out[-1] is not None and out[-1].data is not None


@pytest.mark.asyncio
async def test_bad_frame_length_is_silence_not_an_exception():
    mic = make_mic()
    assert await mic.process(BytesEvent(data=bytes(100))) is None
    assert mic._bad_frame_logged
    # and the module keeps working afterwards
    await open_turn(mic)


@pytest.mark.asyncio
async def test_dump_writes_wav_and_sidecar(tmp_path):
    mic = make_mic(dump_dir=str(tmp_path))
    frames = [silence_frame()] * 3 + [speech_frame()] * 4
    await feed(mic, frames)
    await mic.finalize()

    wavs = list(tmp_path.glob("mic_*.wav"))
    assert len(wavs) == 1
    with wave.open(str(wavs[0]), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == 16000
        assert w.getsampwidth() == 2
        assert w.getnframes() == len(frames) * FRAME
    marks = [json.loads(line) for line in open(str(wavs[0]) + ".jsonl")]
    assert len(marks) == len(frames)
    assert [m["speech"] for m in marks] == [False] * 3 + [True] * 4
    # "in" is the state after the frame: the onset (2nd speech frame) counts
    assert [m["in"] for m in marks] == [False] * 4 + [True] * 3


@pytest.mark.asyncio
async def test_no_dump_without_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("HURI_MIC_DUMP_DIR", raising=False)
    mic = make_mic()
    assert mic._dump.path is None
    await open_turn(mic)
    await mic.finalize()
