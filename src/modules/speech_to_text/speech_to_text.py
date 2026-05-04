from typing import Any, List, Optional

import numpy as np
import whisper
from ray import serve
from ray.serve import handle

from src.core.module import ModuleWithHandle


@serve.deployment
class STTHandle:
    def __init__(
        self,
        model_name: str = "base",
    ):
        super().__init__()

        self.model: whisper.Whisper = whisper.load_model(model_name)

    async def transcribe(self, audio_array: np.ndarray) -> Optional[Any]:
        result: dict = self.model.transcribe(
            audio_array.copy(), condition_on_previous_text=False, fp16=False
        )
        result["text"] = result["text"].strip()
        if not result["text"] or result["text"] == "":
            return None

        return result["text"]


class STT(ModuleWithHandle):
    _handle_cls = STTHandle

    input_type = "voice"
    output_type = "text"

    def __init__(self, handle: handle.DeploymentHandle[STTHandle]):
        super().__init__(handle)

        self.chunks: List[np.ndarray] = []
        self.running = False

    async def process(self, audio: np.ndarray) -> Optional[Any]:
        self.chunks.append(audio)
        if self.running is True:
            return None
        self.running = True
        text = await self.handle.transcribe.remote(np.concatenate(self.chunks, axis=0))
        self.chunks.clear()
        self.running = False
        return text
