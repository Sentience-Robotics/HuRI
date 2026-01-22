import sys
import threading
from dataclasses import dataclass

from src.tools.logger import setup_logger
from typing import Dict
from .zmq.control_channel import Router
from src.core.events import Control, ControlEvent
from .zmq.event_proxy import EventProxy
from .zmq.log_channel import LogPuller
from src.core.agent import AgentConfig


@dataclass
class RouterConfig:
    port: int


@dataclass
class EventProxyConfig:
    xsub: int
    xpub: int


@dataclass
class LogPullerConfig:
    port: int


@dataclass
class HuriConfig:
    hostname: str
    router: RouterConfig
    event_proxy: EventProxyConfig
    log_puller: LogPullerConfig

    @classmethod
    def from_dict(cls, raw: dict):
        return cls(
            hostname=raw["hostname"],
            router=RouterConfig(**raw["router"]),
            event_proxy=EventProxyConfig(**raw["event-proxy"]),
            log_puller=LogPullerConfig(**raw["log-puller"]),
        )


@dataclass
class AgentStatus:
    update_time: int
    config: AgentConfig
    status: Dict[str, int]

    @classmethod
    def from_dict(cls, raw: dict):
        return cls(
            update_time=raw["update_time"],
            config=AgentConfig.from_dict(raw["config"]),
            status=raw["status"],
        )


class HuRI:
    """Wait for Agent to connect, handle module communication and Logging"""

    def __init__(self, config: HuriConfig) -> None:
        self.config = config

        self.router = Router(config.hostname, config.router.port, self._control_handler)
        self.event_proxy = EventProxy(
            config.hostname, "", config.event_proxy.xpub, config.event_proxy.xsub
        )
        self.log_channel = LogPuller(config.hostname, config.log_puller.port)

        self.agents: Dict[bytes, AgentStatus] = {}

        self.stop_event = threading.Event()

        self.logger = setup_logger("HuRI")

    def _control_handler(
        self, identity: bytes, event: ControlEvent
    ) -> bool:  # todo data race ?
        match event.ctrl:
            case Control.REGISTER:
                if event.payload["auth"] != "oui":  # todo wip
                    return False
                self.router.dealers[identity] = True
                self.agents[identity] = AgentStatus.from_dict(**event.payload["agent"])

                self.router.send_control(
                    identity, Control.AUTH_OK, self.config
                )  # todo send all config ?
                self.router.send_control(identity, Control.START)
                return True
            case Control.HEARTBEAT:
                # previous_config = self.router.dealers[identity]
                # self.agents[identity] = todo
                # todo AgentStatus concat
                return True
            case Control.EXITED:
                del self.router.dealers[identity]
                del self.agents[identity]
            case _:
                return False  # todo log

    def run(self) -> None:
        """
        Start LogPuller.
        Start Router.
        Start EventProxy.
        Then loop over RobotShell.cmdloop() to use HuRI commands.
        Then, when exit is requested, call stop()
        """

        "Used to handle log filtering and displaying"
        self.log_channel.start()
        "Used to handle Agent registration and control"
        self.router.start()
        "Used to handle inter-module communication, though events"
        self.event_proxy.start(False, False)

        if not sys.stdin.isatty():
            self.stop_event.wait()
            return

        from src.core.shell import RobotShell

        RobotShell(self).cmdloop()

        self.stop()

    def stop(self) -> None:
        self.router.stop()
        self.event_proxy.stop()
        self.log_channel.stop()
        print("Fully stopped")
