from dataclasses import dataclass

import numpy as np
import struct
from src.core.events import EventData
import logging

_EMAGE_FPS = 30

logger = logging.getLogger("ray.serve")


@dataclass
class Motion(EventData):
    poses: np.ndarray  # (t, 165)  SMPL-X axis-angle, 55 joints × 3
    expressions: np.ndarray  # (t, 100)  facial expression coefficients
    trans: np.ndarray  # (t, 3)    global root translation
    fps: int = _EMAGE_FPS
    pts: float = 0.0  # presentation timestamp in seconds, paired with Audio.pts

    def to_wire(self) -> bytes:
        n_frames = self.poses.shape[0]
        logger.info(
            "Motion frames=%d fps=%d pts=%.3fs",
            n_frames,
            self.fps,
            self.pts,
        )
        header = struct.pack(">dII", self.pts, self.fps, n_frames)
        body = (
            self.poses.astype(np.float32).tobytes()
            + self.expressions.astype(np.float32).tobytes()
            + self.trans.astype(np.float32).tobytes()
        )
        return header + body

    @classmethod
    def from_wire(cls, data: bytes) -> "Motion":
        pts, fps, n_frames = struct.unpack(">dII", data[:16])
        print(f"<< motion: pts={pts:.3f}s frames={n_frames} @ {fps}fps")

        return cls(
            poses=np.ndarray(0),
            expressions=np.ndarray(0),
            trans=np.ndarray(0),
            fps=fps,
            pts=pts,
        )
