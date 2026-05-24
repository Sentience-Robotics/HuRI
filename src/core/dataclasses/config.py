from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional


@dataclass
class ModuleConfig:
    name: str
    args: Mapping[str, Any]

    @classmethod
    def from_dict(self, raw: dict) -> "ModuleConfig":
        return self(
            name=raw["name"],
            args=raw.get("args", {}),
        )


@dataclass
class ClientSenderConfig:
    name: str
    args: Mapping[str, Any]

    @classmethod
    def from_dict(self, raw: dict) -> "ClientSenderConfig":
        return self(
            name=raw["name"],
            args=raw.get("args", {}),
        )


@dataclass
class ClientConfig:
    user_id: Optional[str]
    huri_url: str
    topic_list: List[str]
    senders: Dict[str, ClientSenderConfig]
    modules: Dict[str, ModuleConfig]

    @classmethod
    def from_dict(cls, raw: Dict) -> "ClientConfig":
        senders = {
            sender_id: ClientSenderConfig.from_dict(mod_raw)
            for sender_id, mod_raw in raw.get("senders", {}).items()
        }
        modules = {
            module_id: ModuleConfig.from_dict(mod_raw)
            for module_id, mod_raw in raw.get("modules", {}).items()
        }
        return cls(
            user_id=None,
            huri_url=raw["huri_url"],
            topic_list=raw["topic_list"],
            senders=senders,
            modules=modules,
        )
