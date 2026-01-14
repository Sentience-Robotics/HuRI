import multiprocessing as mp
import signal
import threading
from dataclasses import dataclass
from multiprocessing.synchronize import Event
from typing import Any, Dict, Mapping

from src.core.events import Command, CommandEvent

from src.modules.factory import ModuleFactory
from src.tools.logger import logging, setup_logger

from .huri import HuriConfig
from .zmq.control_channel import Dealer
from .zmq.event_proxy import EventProxy
from .zmq.log_channel import LogPusher


@dataclass
class ForwarderProxyConfig:
    down_xsub: int
    up_xpub: int

    @classmethod
    def from_dict(cls, raw: dict):
        return cls(
            down_xsub=raw["down-xsub"],
            up_xpub=raw["up-xpub"],
        )


@dataclass
class ModuleConfig:
    name: str
    args: Mapping[str, Any]
    logging: int

    @classmethod
    def from_dict(cls, raw: dict):
        level = logging._nameToLevel.get(
            raw.get("logging", "INFO"),
            logging.INFO,
        )
        return cls(
            name=raw["name"],
            args=raw.get("args", {}),
            logging=level,
        )


@dataclass
class AgentConfig:
    id: str
    hostname: str
    huri: HuriConfig
    logging: int
    forwarder_proxy: ForwarderProxyConfig
    modules: Dict[str, ModuleConfig]

    @classmethod
    def from_dict(cls, raw: dict):
        level = logging._nameToLevel.get(
            raw.get("logging", "INFO").upper(),
            logging.INFO,
        )
        modules = {
            module_id: ModuleConfig.from_dict(mod_raw)
            for module_id, mod_raw in raw.get("modules", {}).items()
        }
        return cls(
            id=raw["id"],
            hostname=raw["hostname"],
            huri=HuriConfig.from_dict(raw["huri"]),
            forwarder_proxy=ForwarderProxyConfig.from_dict(raw["forwarder-proxy"]),
            logging=level,
            modules=modules,
        )


class Agent:
    """Control Modules and communication with HuRI"""

    def __init__(self, config: AgentConfig) -> None:
        self.modules: Dict[str, ModuleConfig] = config.modules
        self.config = config

        self.processes: Dict[str, mp.Process] = {}
        self.stop_events: Dict[str, Event] = {}

        self.threads: Dict[str, threading.Thread] = {}

        self.log_pusher = LogPusher(
            hostname=config.huri.hostname, port=config.huri.log_puller.port
        )

        self.dealer = Dealer(
            hostname=config.huri.hostname,
            port=config.huri.router.port,
            executor=self._command_handler,
            logger=setup_logger("Dealer", log_queue=self.log_pusher.log_queue),
        )

        self.up_proxy = EventProxy(
            hostname=config.hostname,
            connect_hostname=config.huri.hostname,
            xpub_port=config.huri.event_proxy.xsub,
            xsub_port=config.forwarder_proxy.up_xpub,
            logger=setup_logger("UpProxy", log_queue=self.log_pusher.log_queue),
        )
        self.down_proxy = EventProxy(
            hostname=config.hostname,
            connect_hostname=config.huri.hostname,
            xpub_port=config.forwarder_proxy.down_xsub,
            xsub_port=config.huri.event_proxy.xpub,
            logger=setup_logger("DownProxy", log_queue=self.log_pusher.log_queue),
        )

        self.logger = setup_logger(
            f"Agent {self.dealer.identity}", log_queue=self.log_pusher.log_queue
        )

    def _command_handler(self, command: CommandEvent) -> bool:
        match command.cmd:
            case Command.START:
                for name in self.modules:
                    self.start_module(name)
                pass
            case Command.STOP:
                for name in list(self.processes.keys()):
                    self.stop_module(name)
            case Command.START_MODULE:
                return self.start_module(**command.payload)
            case Command.STOP_MODULE:
                return self.stop_module(**command.payload)
            case Command.STATUS:
                return self.status()
            case Command.EXIT:
                self.exit()
                pass
            case _:
                return False  # todo log

    @staticmethod
    def _start_module(
        name: str,
        module_config: ModuleConfig,
        agent_config: AgentConfig,
        log_queue: mp.Queue,
        stop_event: Event,
    ) -> None:
        """Helper function to start module in child process."""
        logger = setup_logger(
            module_config.name, level=module_config.logging, log_queue=log_queue
        )

        module = ModuleFactory.create(name, module_config.args)
        module.set_custom_logger(logger)

        def handle_sigint(signum, frame):
            logger.info("Ctrl+C ignored in child module")

        signal.signal(signal.SIGINT, handle_sigint)

        module.start_module(
            agent_config.hostname,
            agent_config.forwarder_proxy.up_xpub,
            agent_config.forwarder_proxy.down_xsub,
            stop_event=stop_event,
        )

    def start_module(self, name) -> None:
        """Check if module is registered and not already running, and start a child process."""
        if name not in self.modules:
            self.logger.warning(
                f"{name} is not in the registered Modules: {self.modules.keys()}"
            )
            return
        if name in self.processes:
            self.logger.warning(
                f"{name} is already running (PID={self.processes[name].pid})"
            )
            return

        module_config = self.modules[name]
        stop_event = mp.Event()
        p = mp.Process(
            target=self._start_module,
            args=(
                name,
                module_config,
                self.config,
                self.log_pusher.log_queue,
                stop_event,
            ),
            daemon=True,
        )
        self.processes[name] = p
        self.stop_events[name] = stop_event
        self.log_pusher.level_filter.add_level(name)

        p.start()
        self.logger.info(f"{name} ({module_config.name}) started (PID={p.pid})")

    def stop_module(self, name) -> None:
        if name in self.processes:
            self.logger.info(f"Stopping {name}...")
            self.stop_events[name].set()
            self.processes[name].join(timeout=5)
            if self.processes[name].is_alive():
                self.logger.warning(f"{name} did not stop in time, killing")
                self.processes[name].kill()
            self.logger.info(f"{name} stopped")
            del self.processes[name]
            del self.stop_events[name]
            self.log_pusher.level_filter.del_level(name)

    def exit(self) -> None:
        for name in list(self.processes.keys()):
            self.stop_module(name)

        self.dealer.stop()
        self.up_proxy.stop()
        self.down_proxy.stop()
        for name, thread in self.threads.items():
            self.logger.info(f"Stopping {name} thread...")
            thread.join(timeout=5)
            self.logger.info(f"{name} thread stopped")
            self.log_pusher.level_filter.del_level(name)

        self.log_pusher.stop()
        print("Fully stopped")

    def status(self) -> None:
        """Print status of all modules and router."""
        print("=== Module Status ===")
        for name in self.modules:
            process = self.processes.get(name)
            if process:
                state = "alive" if process.is_alive() else "stopped"
                print(f"- {name}: {state} (PID={process.pid})")
            else:
                print(f"- {name}: stopped")
        print("=====================")

    def set_root_log_level(self, level: int) -> None:
        self.log_pusher.level_filter.set_root_level(level)

    def set_log_level(self, name: str, level: int) -> None:
        self.log_pusher.level_filter.set_level(name, level)

    def set_log_levels(self, level: int) -> None:
        self.log_pusher.level_filter.set_levels(level)

    def _connect_to_huri(self) -> None:
        self.log_pusher.level_filter.add_level("Dealer")
        self.threads["Dealer"] = threading.Thread(target=self.dealer.start)
        self.threads["Dealer"].start()

    def _start_event_proxies(self) -> None:
        """Used to handle inter-module communication, though events"""
        self.log_pusher.level_filter.add_level("UpProxy")
        self.log_pusher.level_filter.add_level("DownProxy")
        self.threads["UpProxy"] = threading.Thread(
            target=self.up_proxy.start, args=[True, False]
        )
        self.threads["DownProxy"] = threading.Thread(
            target=self.down_proxy.start, args=[False, True]
        )

        self.threads["UpProxy"].start()
        self.threads["DownProxy"].start()

    def run(self) -> None:
        """Start event router and modules"""  # TODO config (also logs levels)

        try:
            self.log_pusher.start()
            self._connect_to_huri()
            self._start_event_proxies()
        except Exception as e:
            self.logger.error(e)
            return

        while True:
            data = input()
            self.down_proxy.publish("std.in", data=data)
