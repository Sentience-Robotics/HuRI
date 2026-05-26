import logging
from typing import Dict, Type

from src.modules.rag.rag import RAG
from src.modules.speech_to_text.microphone_vad import MIC
from src.modules.speech_to_text.speech_to_text import STT
from src.modules.speech_to_text.text_aggregator import TAG

from .factory import Module

_LOG = logging.getLogger(__name__)


def get_modules() -> Dict[str, Type[Module]]:
    modules: Dict[str, Type[Module]] = {"mic": MIC, "stt": STT, "tag": TAG, "rag": RAG}

    # The following imports may contain modules with custom dependencies, depending on the Dockerfile
    # CPU doesn't need some dependencies, nor the AMD that isn't compatible
    try:
        from src.modules.text_to_speech.text_to_speech import TTS
    except Exception as exc:  # noqa: BLE001
        _LOG.info("Skipping TTS module: %s", exc)
    else:
        modules["tts"] = TTS

    try:
        from src.modules.gesture.gesture import Gesture
    except Exception as exc:  # noqa: BLE001
        _LOG.info("Skipping Gesture module: %s", exc)
    else:
        modules["gesture"] = Gesture

    return modules
