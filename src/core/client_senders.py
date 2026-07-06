import asyncio
import json
import struct
from dataclasses import asdict
from typing import Dict, Type

import numpy as np
import sounddevice as sd
import websockets
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

from src.core.events import EventData
from src.modules.rag.events import RAGQuestion
from src.modules.speech_to_text.events import Transcript


class ClientSender:
    """This class abstract sending data to HuRI.

    output_type: is the topic that the ClientSender will send.
    Data structure must match event topic.

    Class derived from ClientSender must implement input_loop,
    and use ClientSender.send to send data to HuRI. It can be EventData or bytes
    """

    output_type: str

    def __init__(self, ws: websockets.ClientConnection):
        self.ws = ws

    async def input_loop(self):
        raise NotImplementedError

    async def send(self, topic: str, data: EventData | bytes):
        packet: str | bytes
        if isinstance(data, EventData):
            packet = json.dumps({"topic": topic, "data": asdict(data)})
        else:
            topic_bytes = topic.encode()

            packet = struct.pack("!H", len(topic_bytes)) + topic_bytes + data

        await self.ws.send(packet)


class AudioSender(ClientSender):
    # Mic frames go out on "audio_in"; the server's "audio" topic is reserved
    # for TTS output streamed back to us (see MIC.input_type).
    output_type = "audio_in"

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
        print("'\\exit' or CTRL+D/C to exit.")
        session: PromptSession = PromptSession()
        try:
            while True:
                with patch_stdout():
                    text = await session.prompt_async(">> ")
                if text == "\\exit":
                    return
                await self.send(
                    self.output_type, RAGQuestion(Transcript(text, True), None)
                )

        except (EOFError, KeyboardInterrupt):
            pass
        finally:
            print("TextSender Exited...")


def get_senders() -> Dict[str, Type[ClientSender]]:
    return {"audio": AudioSender, "text": TextSender}
