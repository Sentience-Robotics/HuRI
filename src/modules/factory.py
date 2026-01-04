from typing import Any, Mapping

from src.core.module import Module

from .rag.mode_controller import ModeController
from .rag.rag import Rag
from .speech_to_text.record_speech import RecordSpeech
from .speech_to_text.speech_to_text import SpeechToText
from .textIO.input import TextInput
from .textIO.output import TextOutput


class ModuleFactory:
    _registry = {}

    @classmethod
    def register(cls, name: str, module_cls):
        cls._registry[name] = module_cls

    @classmethod
    def create(cls, name: str, args: Mapping[str, Any] | None = None) -> Module:
        if name not in cls._registry:
            raise ValueError(f"Unknown module '{name}'")
        return cls._registry[name](**args)


def build_module_factory() -> None:
    ModuleFactory.register("mic", RecordSpeech)
    ModuleFactory.register("stt", SpeechToText)
    ModuleFactory.register("inp", TextInput)
    ModuleFactory.register("out", TextOutput)
    ModuleFactory.register("rag", Rag)
    ModuleFactory.register("mod", ModeController)
