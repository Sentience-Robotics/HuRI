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

    def summarize(self) -> str:
        """Short repr that avoids dumping full numpy arrays into the log."""

        if self.data:
            cls = type(self).__name__
            return f"{cls}(shape={self.data.shape}, dtype={self.data.dtype})"
        else:
            return super().summarize()
