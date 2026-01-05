from src.core.module import Module


class TextInput(Module):
    def set_subscriptions(self):
        self.subscribe("std.in", self.stdin_to_text)
        self.subscribe("std.out", lambda: print(">> ", end="", flush=True))

    def stdin_to_text(self, data):
        print(">> ", end="", flush=True)
        if data == "":
            return
        self.publish("text.in", text=data)

    def run_module(self, stop_event=None):
        print(">> ", end="", flush=True)
        stop_event.wait()
