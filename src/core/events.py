import asyncio
from collections import defaultdict

from .module import Module


class EventGraph:

    def __init__(self):

        self.subscribers = defaultdict(list)

    def register(self, module: Module):
        self.subscribers[module.input_type].append(module)

    async def publish(self, event_topic, data):
        for module in self.subscribers[event_topic]:
            asyncio.create_task(self._run(module, data))

    async def _run(self, module: Module, data):

        result = module.process(data)

        if hasattr(result, "__aiter__"):
            async for item in result:
                if item is None:
                    continue
                await self.publish(module.output_type, item)

        else:
            value = await result
            if value is not None:
                await self.publish(module.output_type, value)
