from dataclasses import dataclass, field

from src.core.events import EventData


@dataclass
class RAGResult(EventData):
    """What RAGHandle returns."""

    answer: str
    sources: list[dict] = field(default_factory=list)
