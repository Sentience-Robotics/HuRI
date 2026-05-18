import argparse
import asyncio
import json
import struct
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict, List, Optional, Type

import numpy as np
import sounddevice as sd
import websockets
from omegaconf import OmegaConf
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

from src.core.dataclasses.config import ClientConfig
from src.core.events import EventData
from src.modules.speech_to_text.events import Sentence


class ClientSender:
    """This class abstract sending data to HuRI.

    output_type: is the topic that the ClientSender will send. Data structure must match event topic.

    Class derived from ClientSender must implement input_loop, and use ClientSender.send to send data to HuRI. It can be EventData or bytes
    """

    output_type: str

    def __init__(self, ws: websockets.ClientConnection):
        self.ws = ws

    async def input_loop(self):
        raise NotImplementedError

    async def send(self, topic: str, data: EventData | bytes):
        if isinstance(data, EventData):
            packet = json.dumps({"topic": topic, "data": asdict(data)})
        else:
            topic_bytes = topic.encode()

            packet = struct.pack("!H", len(topic_bytes)) + topic_bytes + data

        await self.ws.send(packet)


class AudioSender(ClientSender):
    output_type = "audio"

    def __init__(
        self, sample_rate: int = 16000, frame_duration: float = 0.030, **kwargs
    ):
        super().__init__(**kwargs)

        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_duration)

    async def input_loop(self):
        loop = asyncio.get_running_loop()

        queue: asyncio.Queue[np.ndarray] = asyncio.Queue()

        def callback(indata: np.ndarray, frames, time, status):
            loop.call_soon_threadsafe(queue.put_nowait, indata.copy())

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            callback=callback,
            blocksize=self.frame_size,
        ):
            while True:
                chunk = await queue.get()
                await self.send(self.output_type, chunk.tobytes())


class TextSender(ClientSender):
    output_type = "question"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def input_loop(self):
        session = PromptSession()
        while True:
            with patch_stdout():
                text = await session.prompt_async(">> ")

            await self.send(self.output_type, Sentence(text))


def get_senders() -> Dict[str, Type[ClientSender]]:
    return {"audio": AudioSender, "text": TextSender}
