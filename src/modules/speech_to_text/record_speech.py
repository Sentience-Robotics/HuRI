from typing import Optional

import numpy as np

from src.core.module import Module


class MIC(Module):
    input_type = "chunk"
    output_type = "voice"

    def __init__(
        self,
        threshold: int = 0,
    ):
        super().__init__()

        self.THRESHOLD: int = threshold

    async def process(self, data: bytes) -> Optional[np.ndarray]:
        audio_array = np.frombuffer(data, dtype=np.int16)
        if np.abs(audio_array).mean() > self.THRESHOLD:
            audio_array_float = audio_array.astype(np.float32) / 32768.0
            return audio_array_float
        return None
