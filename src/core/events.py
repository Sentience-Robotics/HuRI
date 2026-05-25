import asyncio
from collections import defaultdict
from dataclasses import dataclass

from .module import Module


@dataclass
class EventData:
    """An event data must be derived from this class, and use @dataclass decorator.
    Or they can be bytes."""

    pass


class EventGraph:
    def __init__(self):

        self.subscribers = defaultdict(list)

    def register(self, module: Module):
        self.subscribers[module.input_type].append(module)

    async def publish(self, event_topic, data):
        for module in self.subscribers[event_topic]:
            asyncio.create_task(self._run(module, data))

    async def _run(self, module: Module, data):
        try:
            result = module.process(data)

            if hasattr(result, "__aiter__"):
                try:
                    async for item in result:
                        if item is None:
                            continue
                        await self.publish(module.output_type, item)
                except Exception as e:
                    print(f"[ERROR] async generator in {module}: {e}")

            else:
                try:
                    value = await result
                    if value is not None:
                        await self.publish(module.output_type, value)
                except Exception as e:
                    print(f"[ERROR] coroutine in {module}: {e}")

        except Exception as e:
            print(f"[ERROR] process() call failed in {module}: {e}")
