import time
from typing import Dict

from src.module.module import Module
from src.module.shell import ModuleManager, RobotShell
from src.module.speech_to_text.record_speech import RecordSpeech
from src.module.speech_to_text.speech_to_text import SpeechToText
from src.tools.logger import logging


class TTSModule(Module):
    def __init__(self):
        super().__init__()

    def set_subscriptions(self) -> None:
        self.subscribe("llm.response", self.on_llm_response)

    def on_llm_response(self, msg: str):
        self.logger.debug(f"parle -> {msg}")


class LLMModule(Module):
    def __init__(self):
        super().__init__()

    def set_subscriptions(self) -> None:
        self.subscribe("text.in", self.on_speech)

    def on_speech(self, msg: str):
        reponse = f"Réponse à '{msg}'"
        self.logger.debug(f"{reponse}")
        self.publish("llm.response", reponse)


if __name__ == "__main__":
    modules: Dict[str, Module] = {
        "REC": RecordSpeech(),
        "STT": SpeechToText(),
        "LLM": LLMModule(),
        "TTS": TTSModule(),
    }
    manager = ModuleManager(modules)
    manager.start()
    time.sleep(0.1)
    try:
        RobotShell(manager).cmdloop()
    except KeyboardInterrupt:
        manager.stop_all()
    except Exception as e:
        logging.getLogger().error(e)
