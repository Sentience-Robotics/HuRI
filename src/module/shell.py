import cmd
import logging

from .manager import ModuleManager


class RobotShell(cmd.Cmd):
    intro = "HuRI's shell. Type 'help' to see command's list."
    prompt = "(HuRI) "

    def __init__(self, manager: ModuleManager):
        super().__init__()
        self.manager = manager

    def do_status(self, arg):
        "Display modules and router status."
        self.manager.status()

    def do_start(self, arg):
        "Start a module."
        self.manager.start_module(arg.strip())

    def do_stop(self, arg):
        "Stop a module."
        self.manager.stop_module(arg.strip())

    def do_exit(self, arg):
        "Exit HuRi."
        self.manager.stop_all()
        print("Bye !")
        return True

    def do_log(self, arg):
        """
        Usage:
            log status                 -> display log levels
            log <LEVEL>                -> set global root level (DEBUG/INFO/WARNING/ERROR/CRITICAL), has priority over custom module level
            log <MODULE> <LEVEL>       -> set custom per-module level (module name e.g. STT)
            log all <LEVEL>            -> set all levels (DEBUG/INFO/WARNING/ERROR/CRITICAL)
        """
        parts = arg.strip().split()
        if not parts:
            print("Missing arguments. Type 'help log' for usage.")
            return

        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }

        try:
            if parts[0].lower() == "status":
                self.manager.log_status()
                return

            if len(parts) == 1:
                # Set root level
                level = level_map.get(parts[0].upper())
                if level is None:
                    print(f"Unknown level: {parts[0]}")
                    return
                self.manager.set_root_log_level(level)
                print(f"Root log level set to {parts[0].upper()}")

            elif len(parts) == 2:
                level = level_map.get(parts[1].upper())
                if level is None:
                    print(f"Unknown level: {parts[1]}")
                    return
                if parts[0].lower() == "all":
                    self.manager.set_log_levels(level)
                    print(f"All log levels set to {parts[1].upper()}")
                else:
                    module = parts[0]
                    self.manager.set_log_level(module, level)
                    print(f"{module} log level set to {parts[1].upper()}")

            else:
                print("Invalid arguments. Type 'help log' for usage.")
        except Exception as e:
            print(f"Error setting log level: {e}")
