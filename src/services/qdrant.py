from typing import Any

from src.services.docker_services import QdrantService

def build_qdrant(config: dict) -> Any:
    return QdrantService.bind(  # type: ignore[attr-defined]
        port=config.get("port", 6333),
        image=config.get("image", "qdrant/qdrant:latest"),
        storage_volume=config.get("storage_volume", "qdrant_data"),
    )

