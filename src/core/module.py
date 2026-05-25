from typing import Any, Optional, Type

from ray.serve import handle


class Module:
    input_type: Optional[str]
    output_type: Optional[str]


    async def process(self, _) -> Optional[Any]:
        raise NotImplementedError


class ModuleWithHandle(Module):
    _handle_cls: Type[Any]

    def __init__(self, _handle: handle.DeploymentHandle, **kwargs):
        super().__init__(**kwargs)
        self._handle = _handle


class ModuleWithId(Module):
    def __init__(self, _user_id: str, **kwargs):
        super().__init__(**kwargs)
        self._user_id = _user_id

    def get_user_context(self) -> dict:
        """Override in subclasses to provide user-specific context."""
        return {"_user_id": self._user_id}
