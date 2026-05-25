from dataclasses import asdict

from fastapi import WebSocket

from src.core.events import EventData
from src.core.module import Module


class Sender(Module):
    """Sender Module

    Send output data to the client.
    This data must be JSON serialisable, like a dataclass.

    input: auto, output: None"""

    output_type = None

    def __init__(self, ws: WebSocket, type: str):
        super().__init__()
        self.ws: WebSocket = ws
        self.input_type = type

    async def process(self, data: EventData | bytes):
        if isinstance(data, bytes):
            await self.ws.send_bytes(data)
        elif isinstance(data, EventData):
            await self.ws.send_json(asdict(data))
        else:
            await self.ws.send_text(data)
