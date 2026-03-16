from src.core.module import Module
from src.core.huri import WebSocket
from typing import Any


class Sender(Module):
    """Module to send output data to the client"""

    input_type = ...
    output_type = None

    def __init__(self, ws: WebSocket, type: str):
        super().__init__()
        self.ws: WebSocket = ws
        self.input_type = type

    async def process(self, data: Any):
        await self.ws.send_text(data)
