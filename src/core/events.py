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
    """
    Asynchronous event routing system for HuRI modules.

    The EventGraph is responsible for:
        - Registering module subscribers
        - Routing events between modules
        - Executing module pipelines asynchronously
        - Handling coroutine and async generator outputs

    Modules subscribe to events through their `input_type`.
    When an event is published, all subscribed modules are executed
    concurrently.

    Supports:
        - Coroutine-based modules
        - Async generator streaming modules
        - Recursive event propagation

    :subscribers:
        Dictionary mapping event topics to subscribed modules.
    """

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
