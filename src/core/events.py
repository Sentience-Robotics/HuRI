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


class Control(Enum):
    # Agent -> HuRI
    REGISTER = "REGISTER"  # send auth + agent config
    HEARTBEAT = "HEARTBEAT"  # send agent heartbeat + modified config
    EXITED = "EXITED"  # send exited info
    # HuRI -> Agents
    AUTH_OK = "AUTH_OK"  # send huri config (after)
    START = "START"  # start all modules
    STOP = "STOP"  # stop all modules
    START_MODULE = "START_MODULE"  # start specific modules
    STOP_MODULE = "STOP_MODULE"  # stop specific modules
    EXIT = "EXIT"  # exit agent


@dataclass
class ControlEvent:
    ctrl: Control
    payload: Mapping[str, Any]

    @classmethod
    def from_dict(cls, raw: Dict):
        return cls(ctrl=Control(raw["ctrl"]), payload=raw["payload"])

    def serialize(self) -> Sequence:
        print(self.ctrl.value)

        return [
            self.ctrl.value.encode(),
            json.dumps(self.payload).encode(),
        ]

    @classmethod
    def deserialize(cls, raw: List[bytes]):
        ctrl, payload = raw
        return cls(ctrl=Control(ctrl.decode()), payload=json.loads(payload.decode()))
