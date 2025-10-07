import io
import queue
import threading
import time
from typing import Callable, List, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper

from src.module.module import Event, Module


class SpeechToText(Module):
    def __init__(
        self,
        model_name: str = "base.en",
        device: str = "cpu",
        sample_rate: int = 16000,
    ):
        super().__init__()
        if device == "cpu":
            import warnings

            warnings.filterwarnings(
                "ignore", message="FP16 is not supported on CPU; using FP32 instead"
            )
        self.model: whisper.Whisper = whisper.load_model(model_name, device=device)
        self.SAMPLE_RATE: int = sample_rate
        self.running: bool = False
        self.audio_queue: queue.Queue = queue.Queue()
        self.transcriptions: queue.Queue = queue.Queue()
        self.pause_record = threading.Semaphore(1)
        self.audio_to_process = threading.Semaphore(0)
        self.prompt_available = threading.Semaphore(0)
        self.noise_profile: np.ndarray

    def process_audio(self, buffer: bytes) -> None:
        if not buffer:
            return

        audio_array = np.frombuffer(buffer, dtype=np.int16)
        audio_array = audio_array.astype(np.float32) / 32768.0

        result: dict = self.model.transcribe(audio_array, language="en")
        result["text"] = result["text"].strip()
        if not result["text"] or result["text"] == "":
            return

        self.publish("text.in", result["text"])

    def set_subscriptions(self) -> None:
        self.subscribe("speech.in", self.process_audio)
