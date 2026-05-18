from typing import Any, Dict, List, Mapping, Type

from src.core.dataclasses.config import ModuleConfig
from src.core.events import EventData
from src.core.module import Module, ModuleWithHandle, handle


class EventDataFactory:
    def __init__(self):
        self._registry: Dict[str, Type[EventData | bytes]] = {}

    def register(self, topic: str, event_cls: Type[EventData | bytes] | None) -> None:
        if topic in self._registry:
            if event_cls is None or event_cls == self._registry[topic]:
                return
            else:
                raise RuntimeError(
                    f"event data mismatch: {event_cls} and {self._registry[topic]} for event {topic}"
                )
        if event_cls is None:
            raise RuntimeError(f"event data is not defined for event {topic}")

        self._registry[topic] = event_cls

    def create(self, topic: str, data: Mapping[str, Any] | bytes) -> EventData:
        if topic not in self._registry:
            raise RuntimeError(f"unknown event topic {topic}")

        event_cls = self._registry[topic]

        if issubclass(event_cls, EventData):
            return event_cls(**data)

        return data


class ModuleFactory:
    def __init__(self, handles):
        self._registry: Dict[str, Type[Module]] = {}
        self._handles = handles

    def register(self, name: str, module_cls: Type[Module]) -> None:
        if not issubclass(module_cls, Module):
            raise TypeError(f"{module_cls} must inherit from Module")
        if issubclass(module_cls, ModuleWithHandle):
            if name not in self._handles:
                raise RuntimeError(
                    f"Handles not bound for '{name}'. Check your module config first."
                )
        self._registry[name] = module_cls

    def create(self, name: str, args: Mapping[str, Any] | None = None) -> Module:
        if name not in self._registry:
            raise ValueError(f"Unknown module '{name}'")
        module_cls = self._registry[name]

        if args is None:
            args = {}
        if issubclass(module_cls, ModuleWithHandle):
            if name not in self._handles:
                raise RuntimeError(
                    f"Handles not bound for '{name}'. Check your config first."
                )

            return module_cls(handle=self._handles[name], **args)
        return module_cls(**args)

    def create_from_config(
        self, module_configs: Dict[str, ModuleConfig]
    ) -> List[Module]:
        modules: List[Module] = []
        for module_config in module_configs.values():
            modules.append(self.create(module_config.name, module_config.args))

        if modules == []:
            raise Exception

        return modules


def bind_deployment_handles(
    modules: Dict[str, Type[Module]],
) -> Dict[str, handle.DeploymentHandle]:
    handles: Dict[str, handle.DeploymentHandle] = {}
    for name, module_cls in modules.items():
        if not issubclass(module_cls, ModuleWithHandle):
            continue

        if not hasattr(module_cls, "_handle_cls"):
            raise TypeError(f"{module_cls.__name__} must define _handle_cls")
        handle_cls = module_cls._handle_cls
        handles[name] = handle_cls.bind()

    return handles
