import sys
import threading
from dataclasses import dataclass
from time import sleep

from src.tools.logger import setup_logger

from .zmq.control_channel import Router
from .zmq.event_proxy import EventProxy
from .zmq.log_channel import LogPuller


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


class HuRI:
    """Wait for Agent to connect, handle module communication and Logging"""

    def __init__(self, config: HuriConfig) -> None:
        self.router = Router(config.hostname, config.router.port)
        self.event_proxy = EventProxy(
            config.hostname, "", config.event_proxy.xpub, config.event_proxy.xsub
        )
        self.log_channel = LogPuller(config.hostname, config.log_puller.port)

        self.stop_event = threading.Event()

        self.logger = setup_logger("HuRI")

    def run(self) -> None:
        """
        Start LogPuller.
        Start Router.
        Start EventProxy.
        Then loop over RobotShell.cmdloop() to send input as commandst.
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
