import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer


def get_tts_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return ParlerTTSForConditionalGeneration.from_pretrained(
        "parler-tts/parler-tts-mini-v1"
    ).to(device)


def get_tts_tokenizer():
    return AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")


def tokenize_text(text, tokenizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return tokenizer(text, return_tensors="pt").input_ids.to(device)
