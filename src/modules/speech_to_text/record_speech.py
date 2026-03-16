import queue
import threading
import time
from typing import List, Optional

import numpy as np
import sounddevice as sd

from src.core.module import Module

# from src.core.reactive_layer.IOprocessor import IOprocessor
# from src.core.reactive_layer.IOgestion import IOgestion


class MIC(Module):
    input_type = "chunk"
    output_type = "voice"

    def __init__(
        self,
        threshold: int = 0,
    ):
        super().__init__()

        self.THRESHOLD: int = threshold

    async def process(self, data: bytes) -> np.ndarray:
        audio_array = np.frombuffer(data, dtype=np.int16)
        if np.abs(audio_array).mean() > self.THRESHOLD:
            audio_array = audio_array.astype(np.float32) / 32768.0
            return audio_array
