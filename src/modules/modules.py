from typing import Dict, Type

from src.modules.emotion.emotion_aggregator import EAG
from src.modules.emotion.prosody_analysis import EMO
from src.modules.gesture.gesture import Gesture
from src.modules.rag.question_aggregator import QAG
from src.modules.rag.rag import RAG
from src.modules.speech_to_text.microphone_vad import MIC
from src.modules.speech_to_text.speech_to_text import STT
from src.modules.speech_to_text.text_aggregator import TAG
from src.modules.text_to_speech.text_to_speech import TTS

from .factory import Module


def get_modules() -> Dict[str, Type[Module]]:
    modules: Dict[str, Type[Module]] = {
        "mic": MIC,
        "stt": STT,
        "tag": TAG,
        "emo": EMO,
        "rag": RAG,
        "eag": EAG,
        "qag": QAG,
        "rag": RAG,
        "tts": TTS,
        "gesture": Gesture,
    }
    return modules
