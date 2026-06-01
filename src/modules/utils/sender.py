import logging
import struct
from dataclasses import asdict

import numpy as np
from fastapi import WebSocket

from src.core.events import EventData
from src.core.module import Module
from src.modules.gesture.gesture import Motion
from src.modules.text_to_speech.events import Audio

logger = logging.getLogger("ray.serve")


class Sender(Module):
    """Sender Module

    Send output data to the client.
    This data must be JSON serialisable, like a dataclass.
    Audio wire format:  [4B sample_rate uint32][1B end][8B pts float64][float32 PCM].
    Motion wire format: [8B pts float64][4B fps uint32][4B n_frames uint32]
                        [poses float32 n*165][expressions float32 n*100][trans float32 n*3].

    input: auto, output: None"""

    output_type = None

    def __init__(self, ws: WebSocket, type: str):
        super().__init__()
        self.ws: WebSocket = ws
        self.input_type = type

    async def process(self, _):
        data = _
        logger.info("[Sender:%s] received %s", self.input_type, type(data).__name__)
        if isinstance(data, bytes):
            await self.ws.send_bytes(self._prefix(data))
        elif isinstance(data, Audio):
            logger.info(
                "[Sender:%s] Audio samples=%d sr=%d end=%s pts=%.3fs",
                self.input_type, data.data.shape[0], data.sample_rate, data.end, data.pts,
            )
            header = struct.pack(">IBd", data.sample_rate, int(data.end), data.pts)
            await self.ws.send_bytes(self._prefix(header + data.data.tobytes()))
        elif isinstance(data, Motion):
            n_frames = data.poses.shape[0]
            logger.info(
                "[Sender:%s] Motion frames=%d fps=%d pts=%.3fs",
                self.input_type, n_frames, data.fps, data.pts,
            )
            header = struct.pack(">dII", data.pts, data.fps, n_frames)
            body = (
                data.poses.astype(np.float32).tobytes()
                + data.expressions.astype(np.float32).tobytes()
                + data.trans.astype(np.float32).tobytes()
            )
            await self.ws.send_bytes(self._prefix(header + body))
        elif isinstance(data, EventData):
            await self.ws.send_json({"topic": self.input_type, **asdict(data)})
        else:
            await self.ws.send_text(str(data))

    def _prefix(self, payload: bytes) -> bytes:
        topic_bytes = self.input_type.encode()
        return struct.pack(">H", len(topic_bytes)) + topic_bytes + payload
