import logging
import time
from collections import deque
from typing import Deque, Optional, Tuple

from src.core.module import Module
from src.modules.emotion.events import Emotion
from src.modules.speech_to_text.events import Transcript

from .events import PartialQuestion, RAGQuestion

logger = logging.getLogger("ray.serve")

# A transcript and the emotion of the same turn are both triggered by MIC's
# single end-of-turn marker, so they land within seconds of each other (the
# STT final pass and EMO's last inference). One waiting longer than this for
# its partner is an orphan — its partner failed and was dropped by the bus —
# and must not be paired with the NEXT turn's half.
_MAX_PARTNER_WAIT_S = 30.0


class QAG(Module):
    """QAG Module

    Aggregate sentence and emotion into a RAGQuestion.

    input: partial_question,
    output: question

    Every MIC turn yields exactly one final transcript (TAG, possibly empty)
    and, when the emotion modules are in the session, exactly one emotion
    (EAG) — in either order, and a slow turn's emotion may arrive after the
    next turn's transcript. So both sides are queued and paired FIFO, and an
    empty transcript consumes its emotion so it never gets attached to the
    next real question.

    :use_emotion: default True.
    Set to False if you do not analyze emotion or do not need it.
    """

    input_type = "partial_question"
    output_type = "question"

    def __init__(self, use_emotion: bool = True):
        super().__init__()

        self.use_emotion = use_emotion
        self._transcripts: Deque[Tuple[float, Transcript]] = deque()
        self._emotions: Deque[Tuple[float, Emotion]] = deque()

    async def process(self, partial_question: PartialQuestion) -> Optional[RAGQuestion]:
        now = time.monotonic()
        if partial_question.emotion is not None and self.use_emotion:
            self._emotions.append((now, partial_question.emotion))
        if partial_question.transcript is not None:
            self._transcripts.append((now, partial_question.transcript))

        if self.use_emotion:
            self._drop_orphans(now)

        # One event adds at most one element per side, and after every call at
        # least one queue is empty — so at most one pair completes here (empty
        # pairs are consumed and skipped), which is all process() can return.
        while self._transcripts and (not self.use_emotion or self._emotions):
            _, transcript = self._transcripts.popleft()
            emotion = self._emotions.popleft()[1] if self.use_emotion else None
            if not transcript.text.strip():
                # A turn that transcribed to nothing (noise blip): drop it
                # together with its emotion.
                logger.info("[QAG] empty transcript, turn dropped")
                continue
            return RAGQuestion(transcript, emotion)
        return None

    def _drop_orphans(self, now: float) -> None:
        """Discard a head whose partner never came, so queues can't desync."""
        while (
            self._transcripts
            and not self._emotions
            and now - self._transcripts[0][0] > _MAX_PARTNER_WAIT_S
        ):
            _, t = self._transcripts.popleft()
            logger.warning("[QAG] no emotion arrived for %r, dropped", t.text)
        while (
            self._emotions
            and not self._transcripts
            and now - self._emotions[0][0] > _MAX_PARTNER_WAIT_S
        ):
            self._emotions.popleft()
            logger.warning("[QAG] emotion without transcript, dropped")
