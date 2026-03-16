from .events import EventGraph


class Session:
    def __init__(self, modules):
        self.event_graph = EventGraph()

        for module in modules:
            self.event_graph.register(module)

    async def publish(self, topic, data):
        await self.event_graph.publish(topic, data)
