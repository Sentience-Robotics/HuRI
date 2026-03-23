from typing import Dict, Type

from src.modules.reasoning.embedding import EMB
from src.modules.speech_to_text.record_speech import MIC
from src.modules.speech_to_text.speech_to_text import STT

from .factory import Module


def get_modules() -> Dict[str, Type[Module]]:
    return {"mic": MIC, "stt": STT, "emb": EMB}
