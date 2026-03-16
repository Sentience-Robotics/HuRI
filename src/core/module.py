from typing import Any, Optional


class Module:
    input_type: str
    output_type: str

    async def process(self, _) -> Optional[Any]:
        raise NotImplementedError
