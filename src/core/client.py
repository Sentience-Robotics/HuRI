import asyncio
import json
from dataclasses import asdict
from typing import Dict, List, Type

import websockets

from src.core.dataclasses.config import ClientConfig

from .client_senders import ClientSender, get_senders


class Client:
    """Client is init with a Config, and connects to HuRI using websockets"""

    def __init__(
        self,
        config: ClientConfig,
        senders_dict: Dict[str, Type[ClientSender]] = get_senders(),
    ):
        self.config = config
        self.senders_dict = senders_dict

    async def _receive_loop(self, ws: websockets.ClientConnection):
        while True:
            text = await ws.recv()
            print("<<", text)
            await asyncio.sleep(0.1)

    async def run(self):
        async with websockets.connect(self.config.huri_url) as ws:
            print("Connected to server")

            inputs: List[ClientSender] = [
                self.senders_dict[config.name](ws=ws, **config.args)
                for config in self.config.inputs.values()
            ]

            await ws.send(json.dumps(asdict(self.config)))

            await asyncio.gather(
                *(inp.input_loop() for inp in inputs),
                self._receive_loop(ws),
            )
