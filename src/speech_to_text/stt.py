import os.path

import numpy as np

import whisper

def get_whisper_model(model_name: str):
    if not os.path.exists(f"models/{model_name}"):
        whisper.load_model(model_name)
    return whisper.load_model(model_name)