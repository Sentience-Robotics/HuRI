from typing import Any

from fastapi import WebSocket

from src.core.module import Module


class Sender(Module):
    """Module to send output data to the client"""

    input_type = None
    output_type = None

    def __init__(self, ws: WebSocket, type: str):
        super().__init__()
        self.ws: WebSocket = ws
        self.input_type = type

    async def process(self, data: Any):
        await self.ws.send_text(data)
