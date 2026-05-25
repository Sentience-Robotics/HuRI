import asyncio
import json
import os
from dataclasses import asdict
from typing import Dict, List, Type

import websockets

from src.core.dataclasses.config import ClientConfig
from src.core.user_config import get_or_create_user_id

from .client_senders import ClientSender, get_senders


class Client:
    """Client is init with a Config, and connects to HuRI using websockets"""

    def __init__(
        self,
        config: ClientConfig,
        user_id_file: str | None = None,
        senders_dict: Dict[str, Type[ClientSender]] = get_senders(),
    ):
        self.config = config
        self.user_id_file = user_id_file
        self.senders_dict = senders_dict

    async def _receive_loop(self, ws: websockets.ClientConnection):
        try:
            while True:
                text = await ws.recv()
                print("<<", text)
                await asyncio.sleep(0.1)

        except (asyncio.CancelledError, websockets.ConnectionClosedOK):
            pass

    async def run(self):
        async with websockets.connect(self.config.huri_url) as ws:
            print("Connected to server")

            self.config.user_id = get_or_create_user_id(self.user_id_file)

            senders: List[ClientSender] = [
                self.senders_dict[config.name](ws=ws, **config.args)
                for config in self.config.senders.values()
            ]

            await ws.send(json.dumps(asdict(self.config)))

            init_msg = json.loads(await ws.recv())
            if init_msg.get("type") == "session_init":
                user_id = init_msg["user_id"]
                print(f"Session started with _user_id: {user_id}")

            receive_task = asyncio.create_task(self._receive_loop(ws))
            await asyncio.gather(
                *(sender.input_loop() for sender in senders),
            )

            receive_task.cancel()
