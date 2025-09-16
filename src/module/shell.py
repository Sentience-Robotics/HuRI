import cmd

from .manager import ModuleManager


class RobotShell(cmd.Cmd):
    intro = "HuRI's shell. Type 'help' to see command's list."
    prompt = "(HuRI) "

    def __init__(self, manager: ModuleManager):
        super().__init__()
        self.manager = manager

    def do_status(self, arg):
        self.manager.status()

    def do_start(self, arg):
        self.manager.start(arg.strip())

    def do_stop(self, arg):
        self.manager.stop(arg.strip())

    def do_exit(self, arg):
        self.manager.stop_all()
        print("Bye !")
        return True
