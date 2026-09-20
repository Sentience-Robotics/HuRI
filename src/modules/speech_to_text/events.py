from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.core.events import JsonEvent


@dataclass
class Transcript(JsonEvent):
    text: str
    end: bool


@dataclass
class Voice(JsonEvent):
    data: Optional[np.ndarray]

    def summarize(self) -> str:
        """Short repr that avoids dumping full numpy arrays into the log."""

        # `if self.data:` raises ValueError ("truth value of an array with more
        # than one element is ambiguous") for any real audio frame, and is
        # silently False for a 1-sample one. EventGraph evaluates summarize()
        # on the line *before* it publishes, so that exception dropped every
        # speech frame the VAD produced.
        if self.data is not None:
            cls = type(self).__name__
            return f"{cls}(shape={self.data.shape}, dtype={self.data.dtype})"
        else:
            return super().summarize()
