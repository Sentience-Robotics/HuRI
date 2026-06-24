import asyncio
import os
import wave
from datetime import datetime
from typing import Dict, List, Optional, Type

import numpy as np
import sounddevice as sd
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from scipy.signal import resample

from src.core.client import ClientHook, ClientSender
from src.core.interface import Interface
from src.modules.rag.events import RAGResult
from src.modules.speech_to_text.events import Sentence
from src.modules.text_to_speech.events import Audio


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


class AudioHook(ClientHook[Audio]):
    input_type = Audio

    def __init__(
        self,
        sample_rate=48000,
        incoming_sample_rate=16000,
        save_audio_dir: Optional[str] = None,
        **kwargs,
    ):
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

        # When set, incoming audio chunks are buffered per utterance and written
        # to a .wav under this directory each time an end-of-utterance marker
        # arrives — handy for ear-checking what the TTS actually streamed.
        self.save_audio_dir = save_audio_dir
        self._audio_buf: List[np.ndarray] = []
        self._audio_sr: Optional[int] = None
        self._audio_idx = 0
        if save_audio_dir:
            os.makedirs(save_audio_dir, exist_ok=True)

    def _resample(self, audio: np.ndarray):
        return resample(
            audio,
            int(len(audio) * self.sample_rate / self.incoming_sample_rate),
        ).astype(np.int16)

    def _collect_audio(self, samples: np.ndarray, sample_rate: int, end: bool) -> None:
        if samples.size:
            self._audio_buf.append(samples)
            self._audio_sr = sample_rate
        if end:
            self._flush_audio()

    def _flush_audio(self) -> None:
        if not self._audio_buf or self._audio_sr is None:
            self._audio_buf = []
            return
        audio = np.concatenate(self._audio_buf)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(
            self.save_audio_dir, f"utt-{self._audio_idx:03d}-{stamp}.wav"
        )
        self._write_wav(path, audio, self._audio_sr)
        print(
            f"** saved audio: {path} ({audio.size} samples, "
            f"~{audio.size / self._audio_sr:.2f}s @ {self._audio_sr}Hz)"
        )
        self._audio_idx += 1
        self._audio_buf = []

    @staticmethod
    def _write_wav(path: str, audio: np.ndarray, sample_rate: int) -> None:
        # float32 [-1, 1] -> 16-bit PCM, clipped to avoid wraparound on overshoot.
        pcm = np.clip(audio, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype("<i2")
        with wave.open(path, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm.tobytes())

    async def hook(self, data: Audio):
        print(
            f"<< audio: pts={data.pts:.3f}s "
            f"samples={data.data.size} @ {data.sample_rate}Hz "
            f"end={bool(data.end)}"
        )
        # audio = np.frombuffer(data, dtype=np.int16)

        # audio = self.resample_function(audio)
        # self.stream.write(audio.reshape(-1, 1))

        if self.save_audio_dir:
            self._collect_audio(data.data, data.sample_rate, bool(data.end))


class TextHook(ClientHook[RAGResult]):
    input_type = RAGResult

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def hook(self, data: RAGResult):
        print("<<", data.answer)


class CLIInterface(Interface):
    def __init__(self):
        super().__init__(singletton=None)

    def get_senders(self) -> Dict[str, Type[ClientSender]]:
        return {"audio": AudioSender, "text": TextSender}

    def get_hooks(self) -> Dict[str, Type[ClientHook]]:
        return {"audio": AudioHook, "text": TextHook}


cli_interface = CLIInterface()
