from dataclasses import dataclass, field
from typing import Optional

from src.core.events import EventData
from src.modules.emotion.events import Emotion
from src.modules.speech_to_text.events import Transcript


@dataclass
class RAGResult(EventData):
    """What RAGHandle returns."""

    answer: str
    sources: list[dict] = field(default_factory=list)


@dataclass
class PartialQuestion(EventData):
    """Partial question used to aggregate a sentence to an emotion."""

    transcript: Optional[Transcript]
    emotion: Optional[Emotion]


@dataclass
class RAGQuestion(EventData):
    """Fully aggregated question to send to the RAG."""

    transcript: Transcript
    emotion: Optional[Emotion]
