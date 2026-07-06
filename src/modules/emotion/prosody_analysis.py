import asyncio
from typing import List, Optional

import numpy as np

from src.core.module import Module
from src.modules.speech_to_text.events import Voice

from .events import Emotion


class EMO(Module):
    """EMO Module

    Prosody Analysis of user voice speech.

    input: voice,
    output: emotion

    :model_name: name of the Emotion Analysis model.
    :sample_rate: size of received voice audio. Usually 8000, 16000 or 48000.
    :block_duration: size of received voice audio (in s).
    :analysis_window: duration of audio per analysis (in s).
    """

    input_type = "voice"
    output_type = "emotion"

    def __init__(
        self,
        model_name: str = "superb/hubert-large-superb-er",
        sample_rate: int = 16000,
        block_duration: float = 0.020,  # s
        analysis_window: float = 4.0,  # s
    ):
        super().__init__()

        from transformers import (
            AutoModelForAudioClassification,
            Wav2Vec2FeatureExtractor,
        )

        self.model = AutoModelForAudioClassification.from_pretrained(model_name)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

        self.sample_rate = sample_rate
        self.window_size = int(analysis_window / block_duration)

        self.buffer: List[np.ndarray] = []

        self.silence: bool = True

        self.running = False
        self.lock: asyncio.Lock = asyncio.Lock()

    def _predict_emotion(self, audio_np: np.ndarray):
        import torch

        inputs = self.feature_extractor(
            audio_np, sampling_rate=self.sample_rate, return_tensors="pt", padding=True
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

        predicted_id = int(torch.argmax(probs).item())

        labels = self.model.config.id2label

        return {
            "label": labels[predicted_id],
            "confidence": float(probs[predicted_id]),
            "scores": {labels[i]: float(probs[i]) for i in range(len(labels))},
        }

    async def process(self, voice: Voice) -> Optional[Emotion]:
        if voice.data is None:
            self.silence = True
        else:
            self.silence = False
            async with self.lock:
                self.buffer.append(voice.data)

        async with self.lock:
            if self.running:
                return None
            self.running = True

        async with self.lock:
            buffer_size = len(self.buffer)
            if buffer_size == 0 or (
                self.silence is False and buffer_size < self.window_size
            ):
                self.running = False
                return None
            processing_chunks = self.buffer[: self.window_size]

        processing_audio = np.concatenate(processing_chunks, axis=0)

        emotion_result = await asyncio.to_thread(
            self._predict_emotion, audio_np=processing_audio
        )

        async with self.lock:
            self.buffer = self.buffer[self.window_size :]
            self.running = False

        return Emotion(
            emotion_result["label"],
            emotion_result["confidence"],
            emotion_result["scores"],
            self.silence,
        )
