from typing import List

from .events import EventGraph
from .module import Module


class Session:
    def __init__(self, modules: List[Module]):
        self.event_graph = EventGraph()

        for module in modules:
            self.event_graph.register(module)

    async def publish(self, topic, data):
        await self.event_graph.publish(topic, data)
