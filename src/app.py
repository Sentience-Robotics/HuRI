from ray.serve import Application

from src.core.huri import HuRI
from src.modules.events import get_events
from src.modules.factory import bind_deployment_handles
from src.modules.modules import get_modules


def build_app() -> Application:
    modules = get_modules()
    handles = bind_deployment_handles(modules)
    events = get_events()

    app: Application = HuRI.bind(modules, handles, events)  # type: ignore[attr-defined]
    return app


app = build_app()
