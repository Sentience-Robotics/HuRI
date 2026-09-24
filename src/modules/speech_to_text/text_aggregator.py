import logging
from typing import Optional

from src.core.module import Module
from src.modules.rag.events import PartialQuestion

from .events import Transcript

logger = logging.getLogger("ray.serve")


class TAG(Module):
    """TAG Module

    Forward the final transcript of each utterance as a PartialQuestion.

    Contract with STT (see speech_to_text.py): ``Transcript(end=False)`` is a
    best-effort partial from the sliding window, for live display only;
    ``Transcript(end=True)`` carries the COMPLETE text of the utterance from a
    single whole-utterance pass. So there is nothing left to aggregate here —
    the old character-level merge of overlapping window texts
    (``SequenceMatcher.find_longest_match``) truncated or duplicated words
    whenever two windows transcribed their overlap differently.

    An empty final is forwarded too: QAG uses it to drop the turn together with
    the emotion EAG emitted for it.

    input: transcript,
    output: partial_question
    """

    input_type = "transcript"
    output_type = "partial_question"

    def __init__(
        self,
    ):
        super().__init__()

    async def process(self, transcript: Transcript) -> Optional[PartialQuestion]:
        if not transcript.end:
            return None

        text = transcript.text.strip()
        if text:
            logger.info("[TAG] question: %r", text)
        else:
            logger.info("[TAG] empty final transcript, turn dropped")
        return PartialQuestion(transcript=Transcript(text, True), emotion=None)
