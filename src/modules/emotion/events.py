from dataclasses import dataclass
from typing import Dict

from src.core.events import JsonEvent


@dataclass
class Emotion(JsonEvent):
    label: str
    confidence: float
    scores: Dict[str, float]
    end: bool
