"""MIC -> STT -> TAG through the real EventGraph (one detached task per event)."""

import asyncio

import pytest

from src.core.bus import EventGraph
from src.core.events import BytesEvent
from src.modules.speech_to_text.events import Transcript
from src.modules.speech_to_text.microphone_vad import MIC
from src.modules.speech_to_text.speech_to_text import STT
from src.modules.speech_to_text.text_aggregator import TAG

from .conftest import FakeHandle, FakeVad, silence_frame, speech_frame


class Collector:
    input_type = "partial_question"
    output_type = None

    def __init__(self):
        self.items = []

    async def process(self, data):
        self.items.append(data)
        return None


async def drain(graph_task_count=0):
    for _ in range(20):
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_two_utterances_yield_two_questions_in_order():
    handle = FakeHandle(final_text=lambda call: f"final:{len(call.audio)}")
    mic = MIC(vad_agressiveness=2, silence_duration=1.0, block_duration=0.030)
    mic.vad = FakeVad()
    stt = STT(_handle=handle, language="en", block_duration=0.03)
    tag = TAG()
    collector = Collector()

    graph = EventGraph()
    for module in (mic, stt, tag, collector):
        graph.register(module)

    for _ in range(2):
        for _ in range(100):
            await graph.publish("audio.in", BytesEvent(data=speech_frame()))
        for _ in range(40):
            await graph.publish("audio.in", BytesEvent(data=silence_frame()))
    await drain()

    assert len(collector.items) == 2
    texts = [pq.transcript.text for pq in collector.items]
    assert all(t.startswith("final:") for t in texts)
    assert all(pq.transcript.end for pq in collector.items)
    # Each final saw the whole utterance: 100 speech frames + 33 trailing
    # silence frames (the 34th closes the turn and is not forwarded). The
    # second turn also carries the 6 silence frames left over from the first
    # gap as pre-roll.
    assert len(handle.final_calls[0].audio) == (100 + 33) * 480
    assert len(handle.final_calls[1].audio) == (6 + 100 + 33) * 480
    assert mic.turn == 2


@pytest.mark.asyncio
async def test_mic_closed_mid_utterance_still_yields_the_question():
    """The browser's mic button: the user stops recording mid-sentence, so no
    trailing silence ever reaches MIC. The empty audio.in frame the client
    sends instead must close the turn and run the final pass on everything
    heard so far — and the recording after it must be a turn of its own."""
    handle = FakeHandle(final_text=lambda call: f"final:{len(call.audio)}")
    mic = MIC(vad_agressiveness=2, silence_duration=1.0, block_duration=0.030)
    mic.vad = FakeVad()
    stt = STT(_handle=handle, language="en", block_duration=0.03)
    tag = TAG()
    collector = Collector()

    graph = EventGraph()
    for module in (mic, stt, tag, collector):
        graph.register(module)

    for _ in range(50):
        await graph.publish("audio.in", BytesEvent(data=speech_frame()))
    await graph.publish("audio.in", BytesEvent(data=b""))
    await drain()

    assert len(collector.items) == 1
    assert collector.items[0].transcript == Transcript("final:%d" % (50 * 480), True)
    assert not mic.in_utterance

    # a flush with nothing open changes nothing downstream
    await graph.publish("audio.in", BytesEvent(data=b""))
    await drain()
    assert len(collector.items) == 1

    for _ in range(20):
        await graph.publish("audio.in", BytesEvent(data=speech_frame()))
    await graph.publish("audio.in", BytesEvent(data=b""))
    await drain()
    assert len(collector.items) == 2
    assert collector.items[1].transcript.text == "final:%d" % (20 * 480)
    assert mic.turn == 2


@pytest.mark.asyncio
async def test_mic_closed_flushes_partials_when_final_pass_fails():
    """"Flush everything even if it failed": the forced end must still turn
    into a question when the whole-utterance pass errors out — STT falls back
    to the partials it already has."""
    handle = FakeHandle(partial_text="so far")
    mic = MIC(vad_agressiveness=2, silence_duration=1.0, block_duration=0.030)
    mic.vad = FakeVad()
    stt = STT(_handle=handle, language="en", block_duration=0.03, transcribe_window=0.5)
    tag = TAG()
    collector = Collector()

    graph = EventGraph()
    for module in (mic, stt, tag, collector):
        graph.register(module)

    for _ in range(40):  # > one 0.5 s window: at least one partial lands
        await graph.publish("audio.in", BytesEvent(data=speech_frame()))
    await drain()
    assert stt.partials
    expected = " ".join(stt.partials)
    handle.error = RuntimeError("replica died")
    await graph.publish("audio.in", BytesEvent(data=b""))
    await drain()

    assert [pq.transcript for pq in collector.items] == [Transcript(expected, True)]
    assert len(handle.final_calls) == 1  # the final pass was attempted
    assert not stt.utterance and not stt.partials  # and nothing lingers


@pytest.mark.asyncio
async def test_bus_starts_process_bodies_in_publish_order():
    """The STT invariant (no lock around turn state) relies on the bus starting
    each process() body in publish order and on bodies running until their
    first await."""
    order = []

    class Recorder:
        input_type = "x"
        output_type = None

        async def process(self, data):
            order.append(data.data)
            await asyncio.sleep(0)
            return None

    graph = EventGraph()
    graph.register(Recorder())
    for i in range(50):
        await graph.publish("x", BytesEvent(data=i))
    await drain()
    assert order == list(range(50))
