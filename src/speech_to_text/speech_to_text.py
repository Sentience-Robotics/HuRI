import sounddevice as sd
import numpy as np
import io
import soundfile as sf
import whisper
from typing import List, Optional, Callable
import threading
import queue
import time

class SpeechToText:
    def __init__(
        self,
        model_name: str = "base.en",
        threshold: int = 0,
        silence_duration: float = 1.5,
        chunk_duration: float = 0.5,
        sample_rate: int = 16000,
    ):
        self.model: whisper.Whisper = whisper.load_model(model_name)
        self.THRESHOLD: int = threshold
        self.SILENCE_DURATION: float = silence_duration
        self.CHUNK_DURATION: float = chunk_duration
        self.SAMPLE_RATE: int = sample_rate
        self.running: bool = False
        self.audio_queue: queue.Queue = queue.Queue()
        self.transcriptions: queue.Queue = queue.Queue()
        self.audio_to_process = threading.Semaphore(0)
        self.prompt_available = threading.Semaphore(0)
        self.filter_func: Optional[Callable[[np.ndarray, int], np.ndarray]] = (
            filter_func
        )
        self.noise_profile: np.ndarray

    def process_audio(self, buffer: List[np.ndarray]) -> None:
        if not buffer:
            return

        audio_data: np.ndarray = np.concatenate(buffer, axis=0)
        input_buffer: io.BytesIO = io.BytesIO()
        sf.write(input_buffer, audio_data, self.SAMPLE_RATE, format="WAV")
        # sf.write("test.wav", audio_data, self.SAMPLE_RATE, format="WAV")
        input_buffer.seek(0)
        audio_array, _ = sf.read(input_buffer, dtype="float32")

        result: dict = self.model.transcribe(audio_array, language="en")
        self.transcriptions.put(result["text"])
        self.prompt_available.release()

    def record_audio(self) -> None:
        self.running = True
        while self.running:
            chunk: np.ndarray = self.record_chunk()
            if np.abs(chunk).mean() >= self.THRESHOLD:
                buffer: List[np.ndarray] = [chunk]
                silence_start: Optional[float] = None

                while self.running:
                    chunk = self.record_chunk()
                    buffer.append(chunk)
                    if np.abs(chunk).mean() < self.THRESHOLD:
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start >= self.SILENCE_DURATION:
                            self.audio_queue.put(buffer)
                            self.audio_to_process.release()
                            break
                    else:
                        silence_start = None

    def reduce_noise(self, chunk: np.ndarray) -> np.ndarray:
        return np.clip(chunk - self.noise_profile, -32768, 32767).astype(np.int16)

    def record_chunk(self) -> np.ndarray:
        chunk: np.ndarray = sd.rec(
            int(self.CHUNK_DURATION * self.SAMPLE_RATE),
            samplerate=self.SAMPLE_RATE,
            channels=1,
            dtype="int16",
        )
        sd.wait()
        return self.reduce_noise(chunk)

    def get_prompt(self) -> Optional[str]:
        self.prompt_available.acquire()
        return self.transcriptions.get()

    def calculate_noise_level(self) -> None:
        print("Listening for 10 seconds to calculate noise level...")
        noise_chunk: np.ndarray = sd.rec(
            int(10 * self.SAMPLE_RATE),
            samplerate=self.SAMPLE_RATE,
            channels=1,
            dtype="int16",
        )
        sd.wait()
        self.noise_profile = noise_chunk.mean(axis=0)
        self.THRESHOLD = np.abs(self.reduce_noise(noise_chunk)).mean()
        print(f"Threshold: {self.THRESHOLD}")

    def start(self) -> None:
        if not self.THRESHOLD:
            self.calculate_noise_level()

        self.running = True
        threading.Thread(target=self.record_audio).start()
        threading.Thread(target=self.process_queue).start()

    def process_queue(self) -> None:
        self.audio_to_process.acquire()
        while self.running:
            buffer = self.audio_queue.get()
            self.process_audio(buffer)
            self.audio_to_process.acquire()

    def stop(self) -> None:
        self.running = False
        self.audio_to_process.release()