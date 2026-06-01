import asyncio
import importlib
import json
import os
import struct
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Type

import websockets

from src.core.dataclasses.config import ClientConfig
from src.core.events import EventData

from .interface import Interface


class ClientSender:
    """This class abstract sending data to HuRI.

    output_type: is the event data structure that the ClientSender will send.
    It can be EventData or bytes, and must match event topic it send.

    Class derived from ClientSender must implement input_loop,
    and use ClientSender.send to send data to HuRI.
    """

    output_type: Type[EventData] | bytes

    def __init__(self, topic: str, **_):
        self.topic = topic
        if issubclass(self.output_type, EventData):
            self.send_function = self._send_event_data
        elif issubclass(self.output_type, bytes):
            self.send_function = self._send_bytes
        else:
            raise RuntimeError(f"{self.output_type} should be inherited from \
EventData or bytes")

    async def input_loop(self, ws: websockets.ClientConnection):
        raise NotImplementedError

    async def _send_bytes(self, ws: websockets.ClientConnection, data: bytes):
        topic_bytes = self.topic.encode()
        packet = struct.pack("!H", len(topic_bytes)) + topic_bytes + data

        await ws.send(packet)

    async def _send_event_data(self, ws: websockets.ClientConnection, data: EventData):
        packet = json.dumps({"topic": self.topic, "data": asdict(data)})

        await ws.send(packet)

    async def send(self, ws: websockets.ClientConnection, data: EventData | bytes):
        await self.send_function(ws, data)


class ClientHook:
    """This class abstract processing data from HuRI.

    input_type: is the event data structure that the ClientHook will process.
    It can be EventData or bytes, and must match event topic it react to.

    Class derived from ClientHook must implement hook.

    `singletton` allow hooks to modifies shared ressources,
    and comes from the used interface.
    """

    input_type: Type[EventData] | bytes

    def __init__(self, **_):
        pass

    async def hook(self, singletton: Any, data: EventData | bytes):
        raise NotImplementedError


class Client:
    """Client is init with a Config, and connects to HuRI using websockets"""

    def __init__(
        self,
        config: ClientConfig,
        user_id_file: str = os.path.expanduser("~/.huri_user_id"),
    ):
        self.config = config

        module_path, object_name = self.config.interface_path.split(":", 1)

        module = importlib.import_module(module_path)
        interface: Interface = getattr(module, object_name)

        self.singletton = interface.singletton

        available_senders = interface.get_senders()
        self.senders: List[ClientSender] = [
            available_senders[sender.name](topic=sender.topic, **sender.args)
            for sender in self.config.senders.values()
        ]

        available_hooks = interface.get_hooks()
        self.hooks: Dict[str, List[ClientHook]] = defaultdict(list)
        for hook in self.config.hooks.values():
            for topic in hook.topics:
                self.hooks[topic].append(available_hooks[hook.name](**hook.args))

        self.user_id_file = user_id_file

    def _load_user_id(self) -> Optional[str]:
        if os.path.exists(self.user_id_file):
            with open(self.user_id_file) as f:
                return f.read().strip()
        return None

    def _save_user_id(self, _user_id: str):
        with open(self.user_id_file, "w") as f:
            f.write(_user_id)

    async def _receive_loop(self, ws: websockets.ClientConnection):
        try:
            while True:
                msg = await ws.recv()

                if isinstance(msg, bytes):
                    topic_len = struct.unpack("!H", msg[:2])[0]

                    topic = msg[2 : 2 + topic_len].decode()
                    data = msg[2 + topic_len :]
                else:
                    event = json.loads(msg)
                    topic = event["topic"]
                    data = event["data"]

                for hook in self.hooks[topic]:
                    if not issubclass(hook.input_type, bytes):
                        data = hook.input_type(**data)
                    asyncio.create_task(hook.hook(self.singletton, data))

        except (asyncio.CancelledError, websockets.ConnectionClosedOK):
            pass

    async def run(self):
        async with websockets.connect(self.config.huri_url) as ws:
            print("Connected to server")

            self.config.user_id = self._load_user_id()

            await ws.send(json.dumps(asdict(self.config)))

            init_msg = json.loads(await ws.recv())
            if init_msg.get("type") == "session_init":
                user_id = init_msg["user_id"]
                self._save_user_id(user_id)
                print(f"Session started with _user_id: {user_id}")

            receive_task = asyncio.create_task(self._receive_loop(ws=ws))
            await asyncio.gather(
                *(sender.input_loop(ws=ws) for sender in self.senders),
            )

            receive_task.cancel()
