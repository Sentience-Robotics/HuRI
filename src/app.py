from ray.serve import Application

from src.core.huri import HuRI
from src.modules.factory import bind_deployment_handles
from src.modules.modules import get_modules


def build_app() -> Application:
    modules = get_modules()
    handles = bind_deployment_handles(modules)

    app: Application = HuRI.bind(modules, handles)  # type: ignore[attr-defined]
    return app


app = build_app()
