from typing import Any

from src.services.docker_services import OllamaService

def build_ollama(config: dict) -> Any:
    return OllamaService.options(  # type: ignore[attr-defined]
        num_replicas=config.get("num_replicas", 1),
    ).bind(
        model=config.get("model", "mistral:7b"),
        image=config.get("image", "ollama/ollama:latest"),
        gpu_devices=config.get("gpu_devices", False),
    )
