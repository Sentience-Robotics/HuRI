import multiprocessing as mp
import time
from multiprocessing.synchronize import Event
from typing import Dict, List

from .event_router import EventRouter
from .module import Module


class ModuleManager:
    def __init__(self, modules: Dict[str, Module]):
        self.modules: Dict[str, Module] = modules

        self.processes: Dict[str, mp.Process] = {}
        self.router_process: mp.Process = None
        self.stop_events = {}

    def _start_router(self):
        if self.router_process and self.router_process.is_alive():
            return

        self.router_process = mp.Process(
            target=lambda: EventRouter().start(), daemon=True
        )
        self.router_process.start()
        time.sleep(0.01)
        print(f"[Manager] Router started (PID={self.router_process.pid})")

    @staticmethod
    def _run_module(module: Module, name: str, stop_event: Event):
        module(name).run(stop_event=stop_event)

    def start(self):
        self._start_router()
        for name, mod_cls in self.modules.items():
            if name in self.processes:
                continue
            stop_event = mp.Event()
            p = mp.Process(
                target=self._run_module, args=(mod_cls, name, stop_event), daemon=True
            )
            p.start()
            self.processes[name] = p
            self.stop_events[name] = stop_event
            print(f"[Manager] {name} started (PID={p.pid})")

    def status(self):
        """Print status of all modules and router."""
        print("=== Module Status ===")
        if self.router_process:
            router_state = "alive" if self.router_process.is_alive() else "stopped"
            print(f"- Router: {router_state} (PID={self.router_process.pid})")
        else:
            print("- Router: not started")

        for name in self.modules:
            process = self.processes.get(name)
            if process:
                state = "alive" if process.is_alive() else "stopped"
                print(f"- {name}: {state} (PID={process.pid})")
            else:
                print(f"- {name}: stopped")
        print("=====================")

    def stop(self, name):
        if name in self.processes:
            print(f"[Manager] Stopping {name}...")
            self.stop_events[name].set()
            self.processes[name].join(timeout=5)
            if self.processes[name].is_alive():
                print(f"[Manager] {name} did not stop in time, killing")
                self.processes[name].kill()
            print(f"[Manager] {name} stopped")
            del self.processes[name]
            del self.stop_events[name]

    def stop_all(self):
        for name in list(self.processes.keys()):
            self.stop(name)
        if self.router_process and self.router_process.is_alive():
            print("[Manager] Stopping router...")
            self.router_process.terminate()
            self.router_process.join(timeout=5)
            print("[Manager] Router stopped")
