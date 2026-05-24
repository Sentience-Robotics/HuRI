from typing import Dict, Type

from src.modules.rag.rag import RAG
from src.modules.speech_to_text.microphone_vad import MIC
from src.modules.speech_to_text.speech_to_text import STT
from src.modules.speech_to_text.text_aggregator import TAG

from .factory import Module


def get_modules() -> Dict[str, Type[Module]]:
    return {"mic": MIC, "stt": STT, "tag": TAG, "rag": RAG}
