from difflib import SequenceMatcher
from typing import Optional

from src.core.module import Module
from src.modules.rag.events import PartialQuestion

from .events import Transcript


class TAG(Module):
    """TAG Module

    Aggregate all transcriptions and send when transcript end.

    input: transcript,
    output: partial_question
    """

    input_type = "transcript"
    output_type = "partial_question"

    def __init__(
        self,
    ):
        super().__init__()

        self.sentence: str = ""
        self.prev_index: int = 0

    def _merge(self, current: str, new: str) -> str:
        matcher = SequenceMatcher(None, current, new)
        match = matcher.find_longest_match(0, len(current), 0, len(new))
        if match.a >= self.prev_index:
            self.prev_index = match.a
            return current[: match.a] + new[match.b :]

        self.prev_index = len(current)
        return current + new

    async def process(self, transcript: Transcript) -> Optional[PartialQuestion]:
        text = transcript.text

        if text != "":
            if self.sentence == "":
                self.sentence = text
            else:
                self.sentence = self._merge(self.sentence, text)

        if transcript.end and self.sentence != "":
            transcript = Transcript(self.sentence, True)
            self.sentence = ""
            self.prev_index = 0
            return PartialQuestion(transcript=transcript, emotion=None)
        else:
            return None
