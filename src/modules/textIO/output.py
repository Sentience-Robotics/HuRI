from src.core.module import Module


class TextOutput(Module):
    def set_subscriptions(self) -> None:
        self.subscribe("llm.response", self.print_response)

    def print_response(self, text: str) -> None:
        print(f"\r<< {text}")
        self.publish("std.out")
