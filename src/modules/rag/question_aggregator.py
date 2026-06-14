from typing import Optional

from src.core.module import Module
from src.modules.emotion.events import Emotion
from src.modules.speech_to_text.events import Transcript

from .events import PartialQuestion, RAGQuestion


class QAG(Module):
    """QAG Module

    Aggregate sentence and emotion into a RAGQuestion.

    input: partial_question,
    output: question

    :use_emotion: default True.
    Set to False if you do not analyze emotion or do not need it.
    """

    input_type = "partial_question"
    output_type = "question"

    def __init__(self, use_emotion: bool = True):
        super().__init__()

        self.current_transcript: Optional[Transcript] = None
        self.current_emotion: Optional[Emotion] = None

        self.use_emotion = use_emotion

    async def process(self, partial_question: PartialQuestion) -> Optional[RAGQuestion]:
        if partial_question.emotion is not None:
            self.current_emotion = partial_question.emotion

        if partial_question.transcript is not None:
            self.current_transcript = partial_question.transcript

        if self.current_transcript is not None:
            if self.use_emotion:
                if self.current_emotion is not None:
                    return RAGQuestion(self.current_transcript, self.current_emotion)
            else:
                return RAGQuestion(self.current_transcript, None)

        return None
