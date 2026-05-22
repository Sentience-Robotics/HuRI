# from typing import Any, Optional

# import numpy as np
# from ray import serve
# from ray.serve import handle

# from src.core.module import ModuleWithHandle


# @serve.deployment
# class EMBHandle:
#     def __init__(
#         self,
#         model_name: str = "name",
#     ):
#         super().__init__()

#         self.model = model_name  # TODO MVR load embedding model

#     async def embbed(self, data_to_embed: str) -> Optional[Any]:
#         result = self.model + data_to_embed

#         return result


# class EMB(ModuleWithHandle):
#     _handle_cls = EMBHandle

#     input_type = "toembed"
#     output_type = "embedded"

#     def __init__(self, _handle: handle.DeploymentHandle[EMBHandle]):
#         super().__init__(_handle)

#         self.database = ""

#     async def process(self, data_to_embed: np.ndarray | None) -> Optional[Any]:
#         if data_to_embed:
#             embedded = await self._handle.embbed.remote(data_to_embed)

#         # TODO write embedding
#         return embedded
