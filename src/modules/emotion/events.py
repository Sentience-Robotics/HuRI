from dataclasses import dataclass
from typing import Dict

from src.core.events import EventData


@dataclass
class Emotion(EventData):
    label: str
    confidence: float
    scores: Dict[str, float]
    end: bool
