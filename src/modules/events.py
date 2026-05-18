from typing import Dict, Type

from src.core.events import EventData
from src.modules.speech_to_text.events import Sentence, Transcript, Voice


def get_events() -> Dict[str, Type[EventData | bytes]]:
    return {
        "audio": bytes,
        "voice": Voice,
        "transcript": Transcript,
        "question": Sentence,
    }
