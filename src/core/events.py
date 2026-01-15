import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Sequence


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


class Command(Enum):
    REGISTER = "REGISTER"
    AUTH_OK = "AUTH_OK"
    START = "START"
    STOP = "STOP"
    START_MODULE = "START_MODULE"
    STOP_MODULE = "STOP_MODULE"
    STATUS = "STATUS"
    EXIT = "EXIT"


@dataclass
class CommandEvent:
    cmd: Command
    payload: Mapping[str, Any]

    @classmethod
    def from_dict(cls, raw: Dict):
        return cls(cmd=Command(raw["topic"]), payload=raw["payload"])

    def serialize(self) -> Sequence:
        print(self.cmd.value)

        return [
            self.cmd.value.encode(),
            json.dumps(self.payload).encode(),
        ]

    @classmethod
    def deserialize(cls, raw: List[bytes]):
        cmd, payload = raw
        return cls(cmd=Command(cmd.decode()), payload=json.loads(payload.decode()))
