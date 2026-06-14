from typing import Dict, Type

from src.core.events import EventData
from src.modules.emotion.events import Emotion
from src.modules.rag.events import RAGResult
from src.modules.speech_to_text.events import Sentence, Transcript, Voice
from src.modules.text_to_speech.events import Audio, Token
from src.modules.gesture.events import Motion


def get_events() -> Dict[str, Type[EventData | bytes]]:
    events: Dict[str, Type[EventData | bytes]] = {
        "audio_in": bytes,  # inbound mic frames (raw int16 PCM)
        "audio": bytes,
        "voice": Voice,
        "transcript": Transcript,
        "emotion": Emotion,
        "question": Sentence,
        "token": Token,
        "motion": Motion,
    }

    return events
