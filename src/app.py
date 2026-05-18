from pathlib import Path

import yaml
from ray.serve import Application

from src.core.huri import HuRI
from src.modules.factory import bind_deployment_handles
from src.modules.modules import get_modules
from src.modules.rag.docker_services import OllamaService, QdrantService


def load_services_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config" / "huri.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config.get("services", {})


def build_qdrant(config: dict):
    return QdrantService.bind(
        port=config.get("port", 6333),
        image=config.get("image", "qdrant/qdrant:latest"),
        storage_volume=config.get("storage_volume", "qdrant_data"),
    )


def build_ollama(config: dict):
    return OllamaService.options(
        num_replicas=config.get("num_replicas", 1),
    ).bind(
        model=config.get("model", "mistral:7b"),
        image=config.get("image", "ollama/ollama:latest"),
        gpu_devices=config.get("gpu_devices", False),
    )


def build_app() -> Application:
    modules = get_modules()
    services_config = load_services_config()

    qdrant = build_qdrant(services_config.get("qdrant", {}))
    ollama = build_ollama(services_config.get("ollama", {}))

    handles = bind_deployment_handles(modules, ollama=ollama, qdrant=qdrant)
    app: Application = HuRI.bind(modules, handles)
    return app


app = build_app()
