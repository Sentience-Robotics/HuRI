import queue
import threading
import time
from typing import List, Optional

import numpy as np
import sounddevice as sd

from src.module.module import Event, Module


class RecordSpeech(Module):
    def __init__(
        self,
        threshold: int = 0,
        silence_duration: float = 1.0,
        chunk_duration: float = 0.5,
        sample_rate: int = 16000,
    ):
        super().__init__()

        self.THRESHOLD: int = threshold
        self.SILENCE_DURATION: float = silence_duration
        self.CHUNK_DURATION: float = chunk_duration
        self.SAMPLE_RATE: int = sample_rate
        self.running: bool = False
        self.audio_queue: queue.Queue = queue.Queue()
        self.transcriptions: queue.Queue = queue.Queue()
        self.pause_record = threading.Semaphore(1)
        self.audio_to_process = threading.Semaphore(0)
        self.prompt_available = threading.Semaphore(0)
        self.noise_profile: np.ndarray

    def reduce_noise(self, chunk: np.ndarray) -> np.ndarray:
        if np.abs(chunk).mean() <= self.THRESHOLD:
            return chunk

        return np.clip(chunk - self.noise_profile, -32768, 32767).astype(np.int16)

    def record_chunk(self) -> np.ndarray:
        self.pause_record.acquire()
        chunk: np.ndarray = sd.rec(
            int(self.CHUNK_DURATION * self.SAMPLE_RATE),
            samplerate=self.SAMPLE_RATE,
            channels=1,
            dtype="int16",
        ).ravel()
        sd.wait()
        self.pause_record.release()
        return self.reduce_noise(chunk)

    def calculate_noise_level(self) -> None:
        self.logger.info("Listening for 10 seconds to calculate noise level...")
        noise_chunk: np.ndarray = sd.rec(
            int(10 * self.SAMPLE_RATE),
            samplerate=self.SAMPLE_RATE,
            channels=1,
            dtype="int16",
        ).ravel()
        sd.wait()
        self.noise_profile = noise_chunk.mean(axis=0)
        self.THRESHOLD = np.abs(self.reduce_noise(noise_chunk)).mean()
        self.logger.info(f"Threshold: {self.THRESHOLD}")

    def record_audio(self, starting_chunk, stop_event: Event = None) -> None:
        buffer: List[np.ndarray] = [starting_chunk]
        silence_start: Optional[float] = None

        while stop_event is None or not stop_event.is_set():
            chunk = self.record_chunk()
            buffer.append(chunk)

            if np.abs(chunk).mean() <= self.THRESHOLD:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= self.SILENCE_DURATION:
                    if buffer == []:
                        break
                    speech = np.concatenate(buffer, axis=0)
                    self.publish("speech.in", speech.tobytes(), "bytes")
                    break
            else:
                silence_start = None

    def set_subscriptions(self) -> None:
        self.subscribe("speech.in.pause", self.pause())
        self.subscribe("speech.in.resume", self.pause(False))

    def run_module(self, stop_event: Event = None) -> None:
        if not self.THRESHOLD:
            self.calculate_noise_level()
        else:
            self.noise_profile = np.zeros(
                int(self.CHUNK_DURATION * self.SAMPLE_RATE), dtype=np.int16
            )

        while stop_event is None or not stop_event.is_set():
            chunk: np.ndarray = self.record_chunk()

            if np.abs(chunk).mean() > self.THRESHOLD:
                self.record_audio(chunk, stop_event)

    def pause(self, true: bool = True) -> None:
        if true:
            self.pause_record.acquire()
        else:
            self.pause_record.release()
