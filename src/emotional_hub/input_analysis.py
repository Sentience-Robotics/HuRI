import torch
import numpy as np
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor

MODEL_NAME = "superb/hubert-large-superb-er"
model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)


def predict_emotion(audio_np: np.ndarray, sr=16000):
    if audio_np.dtype != np.float32:
        audio_np = audio_np.astype(np.float32)

    inputs = feature_extractor(
        audio_np, sampling_rate=sr, return_tensors="pt", padding=True
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_id = torch.argmax(logits, dim=-1).item()
    predicted_label = model.config.id2label[predicted_id]

    return predicted_label
