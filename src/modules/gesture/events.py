from dataclasses import dataclass

import numpy as np

from src.core.events import EventData

_EMAGE_FPS = 30


@dataclass
class Motion(EventData):
    poses: np.ndarray  # (t, 165)  SMPL-X axis-angle, 55 joints × 3
    expressions: np.ndarray  # (t, 100)  facial expression coefficients
    trans: np.ndarray  # (t, 3)    global root translation
    fps: int = _EMAGE_FPS
    pts: float = 0.0  # presentation timestamp in seconds, paired with Audio.pts
