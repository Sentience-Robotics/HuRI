import asyncio
import logging
from collections import defaultdict
from typing import Any, AsyncGenerator, Coroutine, cast

from .events import EventData
from .module import Module

logger = logging.getLogger("ray.serve")

# One event per 30 ms mic frame on these topics: logging them at INFO buried
# everything else (2 lines per frame, ~1.7 MB per 10 min). Turn boundaries
# stay visible through the modules' own "[MIC] turn"/"[STT] turn" lines.
_PER_FRAME_TOPICS = frozenset({"audio.in", "voice"})


def _level_for(topic) -> int:
    return logging.DEBUG if topic in _PER_FRAME_TOPICS else logging.INFO


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
        subs = self.subscribers[event_topic]
        logger.log(
            _level_for(event_topic),
            "[GRAPH] publish topic=%r subscribers=%s",
            event_topic,
            [type(m).__name__ for m in subs],
        )
        for module in subs:
            asyncio.create_task(self._run(module, data))

    async def _run(self, module: Module, data):
        try:
            result = module.process(data)

            if hasattr(result, "__aiter__"):
                try:
                    generator = cast(AsyncGenerator[EventData | None, None], result)
                    async for item in generator:
                        if item is None:
                            continue
                        logger.log(
                            _level_for(module.output_type),
                            "[GRAPH] %s -> %r: %s",
                            type(module).__name__,
                            module.output_type,
                            item.summarize(),
                        )
                        await self.publish(module.output_type, item)
                except Exception:
                    logger.exception(
                        "[GRAPH] async generator failed in %s", type(module).__name__
                    )

            else:
                coroutine = cast(Coroutine[Any, Any, EventData | None], result)
                try:
                    value = await coroutine
                    if value is not None:
                        logger.log(
                            _level_for(module.output_type),
                            "[GRAPH] %s -> %r: %s",
                            type(module).__name__,
                            module.output_type,
                            value.summarize(),
                        )
                        await self.publish(module.output_type, value)
                except Exception:
                    logger.exception(
                        "[GRAPH] coroutine failed in %s", type(module).__name__
                    )

        except Exception:
            logger.exception(
                "[GRAPH] process() call failed in %s", type(module).__name__
            )
