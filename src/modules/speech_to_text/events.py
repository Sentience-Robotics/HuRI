from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.core.events import EventData


@dataclass
class Transcript(EventData):
    text: str
    end: bool


@dataclass
class Voice(EventData):
    data: Optional[np.ndarray]


@dataclass
class Sentence(EventData):
    text: str
