from typing import Dict, Type

from src.core.events import EventData
from src.modules.speech_to_text.events import Sentence, Transcript, Voice
from src.modules.text_to_speech.events import Audio, Token


def get_events() -> Dict[str, Type[EventData | bytes]]:
    events: Dict[str, Type[EventData | bytes]] = {
        "audio_in": bytes,  # inbound mic frames (raw int16 PCM)
        "audio": bytes,
        "voice": Voice,
        "transcript": Transcript,
        "question": Sentence,
        "token": Token,
    }

    # Motion lives in the gesture module — only available when EMAGE deps installed.
    try:
        from src.modules.gesture.gesture import Motion
    except Exception:
        pass
    else:
        events["motion"] = Motion

    # TTS output "audio" is an Audio dataclass internally, sent to the client by
    # the Sender's Audio branch (never decoded inbound). Inbound mic frames use
    # the separate "audio_in" topic above. The registry keeps bytes for "audio"
    # only for output-type registration. Keep Audio importable for completeness.
    _ = Audio

    return events
