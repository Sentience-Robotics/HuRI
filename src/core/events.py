from dataclasses import dataclass
from typing import Dict, Sequence, List, Mapping, Any
import json


@dataclass
class ModuleEvent:
    """
    Inter-Module communication event
    Module subscribe to a topic and link a callback.
    The payload must correspond to a mapping of params of the callback.
    """

    topic: str
    payload: Mapping[str, Any]

    @classmethod
    def from_dict(cls, raw: Dict):
        return cls(topic=raw["topic"], payload=raw["payload"])

    def serialize(self) -> Sequence:
        return [self.topic.encode(), json.dumps(self.payload).encode()]

    @classmethod
    def deserialize(cls, raw: List[bytes]):
        topic, payload = raw
        return cls(topic=topic.decode(), payload=json.loads(payload.decode()))
