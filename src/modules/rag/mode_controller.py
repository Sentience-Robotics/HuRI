from enum import Enum

from src.core.module import Module


class Modes(Enum):
    LLM = 0
    CONTEXT = 1
    RAG = 2


class ModeController(Module):
    def __init__(self, default_mode: Modes = Modes.LLM):
        super().__init__()
        self.mode = default_mode

    def switchMode(self, mode: str) -> None:
        self.mode = mode

    def processTextInput(self, text: str):
        if "switch llm" in text.lower():
            self.switchMode(Modes.LLM)
        elif "switch context" in text.lower():
            self.switchMode(Modes.CONTEXT)
        elif "switch rag" in text.lower():
            self.switchMode(Modes.RAG)
        elif "bye bye" in text.lower():
            self.publish("exit", "")  # TODO handle (manager being a module) usefull ?
        elif text.strip() == "":
            return
        else:
            topic = f"{str(self.mode.name).lower()}.in"
            self.publish(topic, text)

    def set_subscriptions(self):
        self.subscribe("text.in", self.processTextInput)
        self.subscribe("mode.switch", self.switchMode)
