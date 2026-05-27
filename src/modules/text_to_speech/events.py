from dataclasses import dataclass

import numpy as np

from src.core.events import EventData


@dataclass
class Token(EventData):
    text: str
    end: bool


@dataclass
class Audio(EventData):
    data: np.ndarray
    sample_rate: int
    end: bool = False
