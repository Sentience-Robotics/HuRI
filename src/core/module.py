from typing import Any, Optional


class Module:
    input_type: Optional[str]
    output_type: Optional[str]

    async def process(self, _) -> Optional[Any]:
        raise NotImplementedError
