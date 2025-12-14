import cmd
import logging
import sys

from src.core.huri import HuRI
from src.core.zmq.control_channel import Command


class RobotShell(cmd.Cmd):
    intro = "HuRI's shell. Type 'help' to see command's list."
    prompt = "(HuRI) "

    def __init__(self, huri: HuRI) -> None:
        super().__init__()
        self.huri = huri

    def do_status(self, arg) -> None:
        "Display modules and router status."
        self.huri.router.send_commands(Command("STATUS", []))

    def do_start(self, arg) -> None:
        "Start a module."
        self.huri.router.send_commands(Command("START", [arg.strip()]))

    def do_stop(self, arg) -> None:
        "Stop a module."
        self.huri.router.send_commands(Command("STOP", [arg.strip()]))

    def do_exit(self, arg) -> None:
        "Exit HuRi."
        self.huri.router.send_commands(Command("EXIT", []))
        print("Bye !")
        return True

    # def do_log(self, arg) -> None:
    #     """
    #     Usage:
    #         log status                 -> display log levels
    #         log <LEVEL>                -> set global root level (DEBUG/INFO/WARNING/ERROR/CRITICAL), has priority over custom module level
    #         log <MODULE> <LEVEL>       -> set custom per-module level (module name e.g. STT)
    #         log all <LEVEL>            -> set all levels (DEBUG/INFO/WARNING/ERROR/CRITICAL)
    #     """
    #     parts = arg.strip().split()
    #     if not parts:
    #         print("Missing arguments. Type 'help log' for usage.")
    #         return

    #     level_map = {
    #         "DEBUG": logging.DEBUG,
    #         "INFO": logging.INFO,
    #         "WARNING": logging.WARNING,
    #         "ERROR": logging.ERROR,
    #         "CRITICAL": logging.CRITICAL,
    #     }

    #     try:
    #         if parts[0].lower() == "status":
    #             self.huri.lo.log_status()
    #             return

    #         if len(parts) == 1:
    #             # Set root level
    #             level = level_map.get(parts[0].upper())
    #             if level is None:
    #                 print(f"Unknown level: {parts[0]}")
    #                 return
    #             self.huri.lo.set_root_log_level(level)
    #             print(f"Root log level set to {parts[0].upper()}")

    #         elif len(parts) == 2:
    #             level = level_map.get(parts[1].upper())
    #             if level is None:
    #                 print(f"Unknown level: {parts[1]}")
    #                 return
    #             if parts[0].lower() == "all":
    #                 self.huri.lo.set_log_levels(level)
    #                 print(f"All log levels set to {parts[1].upper()}")
    #             else:
    #                 module = parts[0]
    #                 self.huri.lo.set_log_level(module, level)
    #                 print(f"{module} log level set to {parts[1].upper()}")

    #         else:
    #             print("Invalid arguments. Type 'help log' for usage.")
    #     except Exception as e:
    #         print(f"Error setting log level: {e}")

    # def do_stdin(self, arg) -> None:
    #     "Will disable shell and get lines from stdin to send as 'text.in' event. Exit with CTRL+D."
    #     print("Press CTRL+D to exit...\n>> ", end="")
    #     data = sys.stdin.readline()
    #     while data != "":
    #         self.huri.publish("text.in", data)
    #         print(">> ", end="")
    #         data = sys.stdin.readline()
