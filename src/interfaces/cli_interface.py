import asyncio
from typing import Dict, Type

import numpy as np
import sounddevice as sd
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from scipy.signal import resample

from src.core.client import ClientHook, ClientSender
from src.core.interface import Interface
from src.modules.speech_to_text.events import Sentence


class AudioSender(ClientSender[bytes]):
    def __init__(
        self, sample_rate: int = 16000, frame_duration: float = 0.030, **kwargs
    ):
        super().__init__(**kwargs)

        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_duration)

    async def input_loop(self, ws):
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
                await self.send(ws, chunk.tobytes())


class TextSender(ClientSender[Sentence]):
    output_type = Sentence

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def input_loop(self, ws):
        print("'\\exit' or CTRL+D/C to exit.")
        session: PromptSession = PromptSession()
        try:
            while True:
                with patch_stdout():
                    text = await session.prompt_async(">> ")
                if text == "\\exit":
                    return
                await self.send(ws, Sentence(text))

        except (EOFError, KeyboardInterrupt):
            pass
        finally:
            print("TextSender Exited...")


class AudioHook(ClientHook[bytes]):
    input_type = bytes

    def __init__(self, sample_rate=48000, incoming_sample_rate=16000, **kwargs):
        super().__init__(**kwargs)

        print("Speaker:", sd.query_devices(kind="output"))

        self.incoming_sample_rate = incoming_sample_rate
        self.sample_rate = sample_rate
        self.stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
        )
        self.stream.start()

        self.resample_function = (
            self._resample if sample_rate != incoming_sample_rate else lambda x: x
        )

    def _resample(self, audio: np.ndarray):
        return resample(
            audio,
            int(len(audio) * self.sample_rate / self.incoming_sample_rate),
        ).astype(np.int16)

    async def hook(self, singletton: None, data: bytes):
        audio = np.frombuffer(data, dtype=np.int16)

        audio = self.resample_function(audio)

        self.stream.write(audio.reshape(-1, 1))


class TextHook(ClientHook[Sentence]):
    input_type = Sentence

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def hook(self, singletton: None, data: Sentence):
        print("<<", data.text)


class CLIInterface(Interface):
    def __init__(self):
        super().__init__(singletton=None)

    def get_senders(self) -> Dict[str, Type[ClientSender]]:
        return {"audio": AudioSender, "text": TextSender}

    def get_hooks(self) -> Dict[str, Type[ClientHook]]:
        return {"audio": AudioHook, "text": TextHook}


cli_interface = CLIInterface()
