from typing import Any, Optional, Type

from ray.serve import handle


class Module:
    input_type: str
    output_type: Optional[str]

    async def process(self, _) -> Optional[Any]:
        raise NotImplementedError


class ModuleWithHandle(Module):
    _handle_cls: Type[Any]

    def __init__(self, handle: handle.DeploymentHandle):
        super().__init__()
        self.handle = handle
