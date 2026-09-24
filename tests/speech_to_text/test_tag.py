import pytest

from src.modules.emotion.events import Emotion
from src.modules.rag.events import PartialQuestion
from src.modules.rag.question_aggregator import QAG
from src.modules.speech_to_text.events import Transcript
from src.modules.speech_to_text.text_aggregator import TAG


def emotion(label="hap"):
    return Emotion(label, 0.9, {label: 0.9}, True)


@pytest.mark.asyncio
async def test_tag_emits_only_on_end():
    tag = TAG()
    assert await tag.process(Transcript("hel", False)) is None
    assert await tag.process(Transcript("hello wor", False)) is None
    result = await tag.process(Transcript(" hello world ", True))
    assert result == PartialQuestion(transcript=Transcript("hello world", True), emotion=None)


@pytest.mark.asyncio
async def test_tag_forwards_empty_final():
    tag = TAG()
    result = await tag.process(Transcript("", True))
    assert result == PartialQuestion(transcript=Transcript("", True), emotion=None)


@pytest.mark.asyncio
async def test_tag_is_stateless_across_turns():
    tag = TAG()
    first = await tag.process(Transcript("one", True))
    second = await tag.process(Transcript("two", True))
    assert first.transcript.text == "one"
    assert second.transcript.text == "two"


# ---------------------------------------------------------------------------
# QAG: FIFO pairing; an empty turn must not eat a real question nor park its
# emotion for the next one
# ---------------------------------------------------------------------------


def pq_t(text):
    return PartialQuestion(transcript=Transcript(text, True), emotion=None)


def pq_e(label):
    return PartialQuestion(transcript=None, emotion=emotion(label))


@pytest.mark.asyncio
async def test_qag_pairs_transcript_with_emotion_either_order():
    qag = QAG()
    assert await qag.process(pq_t("q1")) is None
    q1 = await qag.process(pq_e("hap"))
    assert q1.transcript.text == "q1" and q1.emotion.label == "hap"

    assert await qag.process(pq_e("sad")) is None
    q2 = await qag.process(pq_t("q2"))
    assert q2.transcript.text == "q2" and q2.emotion.label == "sad"


@pytest.mark.asyncio
async def test_qag_empty_turn_does_not_eat_a_parked_question():
    """The PC session of 2026-09-11 11:31: 'Hello… Mr. Mouseman' (T1) waits
    for its emotion; a noise blip's empty final (T2) arrives first, then E1,
    then E2. T1 must go out with E1; T2/E2 vanish."""
    qag = QAG()
    assert await qag.process(pq_t("Hello, Mr. Mouseman")) is None
    assert await qag.process(pq_t("")) is None
    q = await qag.process(pq_e("hap"))
    assert q.transcript.text == "Hello, Mr. Mouseman" and q.emotion.label == "hap"
    assert await qag.process(pq_e("sad")) is None  # consumed by the empty turn
    # next real turn pairs with its own emotion
    assert await qag.process(pq_t("next")) is None
    q = await qag.process(pq_e("neu"))
    assert q.transcript.text == "next" and q.emotion.label == "neu"


@pytest.mark.asyncio
async def test_qag_empty_transcript_then_emotion_drops_both():
    qag = QAG()
    assert await qag.process(pq_t("")) is None
    assert await qag.process(pq_e("ang")) is None
    assert await qag.process(pq_t("next")) is None
    q = await qag.process(pq_e("hap"))
    assert q.transcript.text == "next" and q.emotion.label == "hap"


@pytest.mark.asyncio
async def test_qag_emotion_then_empty_transcript_drops_both():
    qag = QAG()
    assert await qag.process(pq_e("ang")) is None
    assert await qag.process(pq_t("  ")) is None
    assert not qag._transcripts and not qag._emotions


@pytest.mark.asyncio
async def test_qag_late_emotion_after_next_turn_transcript():
    """E1 arrives after T2 (slow EMO): T1/E1 then T2/E2, in order."""
    qag = QAG()
    assert await qag.process(pq_t("one")) is None
    assert await qag.process(pq_t("two")) is None
    q1 = await qag.process(pq_e("hap"))
    assert q1.transcript.text == "one" and q1.emotion.label == "hap"
    q2 = await qag.process(pq_e("sad"))
    assert q2.transcript.text == "two" and q2.emotion.label == "sad"


@pytest.mark.asyncio
async def test_qag_orphan_transcript_is_dropped_not_mispaired(monkeypatch):
    import src.modules.rag.question_aggregator as mod

    clock = [1000.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock[0])
    qag = QAG()
    assert await qag.process(pq_t("lost")) is None  # its emotion never comes
    clock[0] += 60
    assert await qag.process(pq_t("fresh")) is None
    q = await qag.process(pq_e("hap"))
    assert q.transcript.text == "fresh" and q.emotion.label == "hap"


@pytest.mark.asyncio
async def test_qag_without_emotion_drops_empty_and_emits_text():
    qag = QAG(use_emotion=False)
    assert await qag.process(pq_t("")) is None
    q = await qag.process(pq_t("hi"))
    assert q.transcript.text == "hi" and q.emotion is None
    # emotions are ignored entirely
    assert await qag.process(pq_e("hap")) is None
    assert not qag._emotions
