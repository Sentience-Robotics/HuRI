import asyncio
import queue
import threading
from typing import Optional

import numpy as np
import whisper
from ray import serve
from ray.serve import handle

from src.core.module import Module


@serve.deployment(num_replicas=5)
class STTHandle:
    def __init__(
        self,
        model_name: str = "base",
    ):
        super().__init__()

        self.model: whisper.Whisper = whisper.load_model(model_name)

    async def process(self, audio_array: np.ndarray) -> Optional[str]:
        result: dict = self.model.transcribe(
            audio_array.copy(), condition_on_previous_text=False, fp16=False
        )
        result["text"] = result["text"].strip()
        if not result["text"] or result["text"] == "":
            return None

        return result["text"]


class STT(Module):
    input_type = "voice"
    output_type = "text"

    def __init__(self, stt_handle: handle.DeploymentHandle[STTHandle]):
        self.stt = stt_handle

        self.chunks = []
        self.running = False

    async def process(self, audio: np.ndarray) -> Optional[str]:
        self.chunks.append(audio)
        if self.running is True:
            return None
        self.running = True
        text = await self.stt.process.remote(np.concatenate(self.chunks, axis=0))
        self.chunks.clear()
        self.running = False
        print(text)
        return text
