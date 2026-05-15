from ray.serve import Application

from src.core.huri import HuRI
from src.modules.factory import bind_deployment_handles
from src.modules.modules import get_modules
from src.modules.rag.docker_services import OllamaService, QdrantService


def build_app() -> Application:
    modules = get_modules()

    qdrant = QdrantService.bind(port=6333)
    ollama = OllamaService.options(num_replicas=1).bind(
        model="mistral:7b",
        image="ollama/ollama:rocm",
        gpu_devices=True,
    )

    handles = bind_deployment_handles(modules, ollama=ollama, qdrant=qdrant)
    app: Application = HuRI.bind(modules, handles)
    return app


app = build_app()
