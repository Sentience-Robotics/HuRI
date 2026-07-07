from typing import Dict, Type

from src.core.events import EventData
from src.modules.emotion.events import Emotion
from src.modules.gesture.events import Motion
from src.modules.rag.events import PartialQuestion, RAGQuestion
from src.modules.speech_to_text.events import Transcript, Voice
from src.modules.text_to_speech.events import Token


def get_events() -> Dict[str, Type[EventData | bytes]]:
    events: Dict[str, Type[EventData | bytes]] = {
        "audio_in": bytes,  # inbound mic frames (raw int16 PCM)
        "audio": bytes,
        "voice": Voice,
        "transcript": Transcript,
        "emotion": Emotion,
        "partial_question": PartialQuestion,
        "question": RAGQuestion,
        "token": Token,
        "motion": Motion,
    }

    return events
