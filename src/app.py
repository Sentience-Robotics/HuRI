from pathlib import Path
from typing import Any

import yaml
from ray.serve import Application

from src.core.huri import HuRI
from src.modules.events import get_events
from src.modules.factory import bind_deployment_handles
from src.modules.modules import get_modules
from src.modules.services.ollama import build_ollama
from src.modules.services.qdrant import build_qdrant

def load_services_config() -> Any:
    config_path = Path(__file__).resolve().parents[1] / "config" / "huri.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config.get("services", {})


def build_app() -> Application:
    modules = get_modules()
    events = get_events()

    services_config = load_services_config()

    qdrant = build_qdrant(services_config.get("qdrant", {}))
    ollama = build_ollama(services_config.get("ollama", {}))

    handles = bind_deployment_handles(modules, ollama=ollama, qdrant=qdrant)
    app: Application = HuRI.bind(modules, handles, events)  # type: ignore[attr-defined]
    return app


app = build_app()
