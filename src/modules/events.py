from typing import Dict, Type

from src.core.events import EventData
from src.modules.gesture.events import Motion
from src.modules.speech_to_text.events import Sentence, Transcript, Voice
from src.modules.text_to_speech.events import Token


def get_events() -> Dict[str, Type[EventData | bytes]]:
    events: Dict[str, Type[EventData | bytes]] = {
        "audio_in": bytes,  # inbound mic frames (raw int16 PCM)
        "audio": bytes,
        "voice": Voice,
        "transcript": Transcript,
        "question": Sentence,
        "token": Token,
        "motion": Motion,
    }

    return events
