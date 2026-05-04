from typing import Any, Dict, List, Mapping, Type

from src.core.dataclasses.config import ModuleConfig
from src.core.module import Module, ModuleWithHandle, handle


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
        for _, module_config in module_configs.items():
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
