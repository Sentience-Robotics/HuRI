import cmd

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
