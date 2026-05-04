from typing import Dict, Type

from src.modules.speech_to_text.record_speech import MIC
from src.modules.speech_to_text.speech_to_text import STT
from src.modules.speech_to_text.text_aggregator import TAG

from .factory import Module


def get_modules() -> Dict[str, Type[Module]]:
    return {"mic": MIC, "stt": STT, "tag": TAG}
