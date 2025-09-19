import time
from multiprocessing.synchronize import Event
from typing import Dict

from src.module.module import Module
from src.module.shell import ModuleManager, RobotShell
from src.tools.logger import logging


class TTSModule(Module):
    def __init__(self, name="TTS", logger=None):
        super().__init__(name, logger)
        self.subscribe("llm.response", self.on_llm_response)

    def on_llm_response(self, msg):
        self.logger.debug(f"[{self.name}] parle -> {msg}")


class LLMModule(Module):
    def __init__(self, name="LLM", logger=None):
        super().__init__(name, logger)
        self.subscribe("speech.in", self.on_speech)

    def on_speech(self, msg):
        reponse = f"Réponse à '{msg}'"
        self.logger.debug(f"[{self.name}] -> {reponse}")
        self.publish("llm.response", reponse)


class STTModule(Module):
    def __init__(self, name="STT", logger=None):
        super().__init__(name, logger)

    def loop(self, stop_event: Event = None):
        i = 0
        while stop_event is None or not stop_event.is_set():
            phrase = f"Phrase numéro {i}"
            self.logger.debug(f"[{self.name}] -> {phrase}")
            self.publish("speech.in", phrase)
            i += 1
            time.sleep(2)


if __name__ == "__main__":
    modules: Dict[str, Module] = {"STT": STTModule, "TTS": TTSModule, "LLM": LLMModule}
    manager = ModuleManager(modules)
    manager.start()
    time.sleep(0.1)
    try:
        RobotShell(manager).cmdloop()
    except KeyboardInterrupt:
        manager.stop_all()
    except Exception as e:
        logging.getLogger().error(e)
