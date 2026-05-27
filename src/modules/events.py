from typing import Dict, Type

from src.core.events import EventData
from src.modules.speech_to_text.events import Sentence, Transcript, Voice
from src.modules.text_to_speech.events import Audio, Token


def get_events() -> Dict[str, Type[EventData | bytes]]:
    events: Dict[str, Type[EventData | bytes]] = {
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

    # TTS output "audio" is an Audio dataclass internally; the websocket boundary
    # only ever decodes raw bytes for the "audio" topic (mic input), so the
    # registry keeps bytes there. Keep Audio importable for type completeness.
    _ = Audio

    return events
