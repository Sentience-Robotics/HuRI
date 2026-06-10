import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .module import Module

logger = logging.getLogger("ray.serve")


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
        subs = self.subscribers[event_topic]
        if event_topic not in ("audio_in",):  # skip mic-frame spam
            logger.info(
                "[GRAPH] publish topic=%r subscribers=%s",
                event_topic, [type(m).__name__ for m in subs],
            )
        for module in subs:
            asyncio.create_task(self._run(module, data))

    async def _run(self, module: Module, data):
        try:
            result = module.process(data)

            if hasattr(result, "__aiter__"):
                try:
                    async for item in result:
                        if item is None:
                            continue
                        logger.info(
                            "[GRAPH] %s -> %r: %s",
                            type(module).__name__, module.output_type, _summarize(item),
                        )
                        await self.publish(module.output_type, item)
                except Exception:
                    logger.exception("[GRAPH] async generator failed in %s", type(module).__name__)

            else:
                try:
                    value = await result
                    if value is not None:
                        logger.info(
                            "[GRAPH] %s -> %r: %s",
                            type(module).__name__, module.output_type, _summarize(value),
                        )
                        await self.publish(module.output_type, value)
                except Exception:
                    logger.exception("[GRAPH] coroutine failed in %s", type(module).__name__)

        except Exception:
            logger.exception("[GRAPH] process() call failed in %s", type(module).__name__)


def _summarize(item) -> str:
    """Short repr that avoids dumping full numpy arrays into the log."""
    cls = type(item).__name__
    data = getattr(item, "data", None)
    if isinstance(data, np.ndarray):
        return f"{cls}(shape={data.shape}, dtype={data.dtype})"
    return f"{cls}({item!r})"
