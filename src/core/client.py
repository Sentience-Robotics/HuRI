import asyncio
import importlib
import json
import os
import struct
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

import numpy as np
import websockets

from src.core.dataclasses.config import ClientConfig
from src.core.events import EventData

T = TypeVar("T", bound=EventData | bytes)


class ClientSender(Generic[T]):
    """This class abstract sending data to HuRI.

    output_type: is the event data structure that the ClientSender will send.
    It can be EventData or bytes, and must match event topic it send.

    Class derived from ClientSender must implement input_loop,
    and use ClientSender.send to send data to HuRI.

    `singletton` is available to access shared ressources.
    """

    output_type: Type[T]

    def __init__(self, topic: str, singletton: Any, **_):
        """
        :topic: topic sent to HuRI
        :singletton: allow to get shared ressources"""
        self.topic = topic
        self.singletton = singletton

    async def input_loop(self, ws: websockets.ClientConnection):
        raise NotImplementedError

    async def _send_bytes(self, ws: websockets.ClientConnection, data: bytes):
        topic_bytes = self.topic.encode()
        packet = struct.pack(">H", len(topic_bytes)) + topic_bytes + data

        await ws.send(packet)

    async def _send_event_data(self, ws: websockets.ClientConnection, data: EventData):
        packet = json.dumps({"topic": self.topic, "data": asdict(data)})

        await ws.send(packet)

    async def send(self, ws: websockets.ClientConnection, data: T):
        if isinstance(data, bytes):
            await self._send_bytes(ws, data)
        else:
            await self._send_event_data(ws, data)


class ClientHook(Generic[T]):
    """This class abstract processing data from HuRI.

    input_type: is the event data structure that the ClientHook will process.
    It can be EventData or bytes, and must match event topic it react to.

    Class derived from ClientHook must implement hook.

    `singletton` is available to access and modifies shared ressources.
    """

    input_type: Type[T]

    def __init__(self, singletton: Any, **_):
        self.singletton = singletton

    async def hook(self, data: T):
        raise NotImplementedError


class Client:
    """Client is init with a Config, and connects to HuRI using websockets"""

    def __init__(
        self,
        config: ClientConfig,
    ):
        self.config = config

        module_path, object_name = self.config.interface_path.split(":", 1)

        module = importlib.import_module(module_path)
        interface = getattr(module, object_name)

        available_senders = interface.get_senders()
        self.senders: List[ClientSender] = [
            available_senders[sender.name](
                topic=sender.topic, singletton=interface.singletton, **sender.args
            )
            for sender in self.config.senders.values()
        ]

        available_hooks = interface.get_hooks()
        self.hooks: Dict[str, List[ClientHook]] = defaultdict(list)
        for hook in self.config.hooks.values():
            for topic in hook.topics:
                self.hooks[topic].append(
                    available_hooks[hook.name](
                        singletton=interface.singletton, **hook.args
                    )
                )

    async def _receive_loop(self, ws: websockets.ClientConnection):
        try:
            while True:
                msg = await ws.recv()

                if isinstance(msg, bytes):
                    topic_len = struct.unpack(">H", msg[:2])[0]

                    topic = msg[2 : 2 + topic_len].decode()
                    data = msg[2 + topic_len :]

                    if topic == "audio" and len(data) >= 13:
                        sample_rate, end, pts = struct.unpack(">IBd", data[:13])
                        # Samples are native-endian float32 (Sender uses ndarray.tobytes()).
                        samples = np.frombuffer(data[13:], dtype=np.float32)
                        data = {
                            "sample_rate": sample_rate,
                            "end": end,
                            "pts": pts,
                            "data": samples,
                        }
                    elif topic == "motion" and len(data) >= 16:
                        pts, fps, n_frames = struct.unpack(">dII", data[:16])
                        print(f"<< motion: pts={pts:.3f}s frames={n_frames} @ {fps}fps")
                        data = {
                            "poses": np.ndarray(),
                            "expressions": np.ndarray(),
                            "trans": np.ndarray(),
                            "fps": fps,
                            "pts": pts,
                        }
                    else:
                        print(f"<< {topic}: bytes ({len(data)}B)")
                else:
                    event = json.loads(msg)
                    topic = event["topic"]
                    data = event["data"]

                for hook in self.hooks[topic]:
                    if not isinstance(data, bytes):
                        data = hook.input_type(**data)
                    asyncio.create_task(hook.hook(data))

        except (asyncio.CancelledError, websockets.ConnectionClosedOK):
            pass

    async def run(self):
        async with websockets.connect(self.config.huri_url) as ws:
            print("Connected to server")

            await ws.send(json.dumps(asdict(self.config)))

            init_msg = json.loads(await ws.recv())
            if init_msg.get("type") == "session_init":
                user_id = init_msg["user_id"]
                print(f"Session started with _user_id: {user_id}")

            receive_task = asyncio.create_task(self._receive_loop(ws=ws))
            await asyncio.gather(
                *(sender.input_loop(ws=ws) for sender in self.senders),
            )

            receive_task.cancel()
