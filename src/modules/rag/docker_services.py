"""
Docker Services for HuRI — Ray-managed Docker containers.

WHAT THIS DOES:
    Each service is a Ray Serve deployment that:
    1. Starts a Docker container when the deployment initializes
    2. Exposes methods to interact with that container
    3. Cleans up the container when the deployment is destroyed

WHY:
    Instead of manually running `docker run ...` before starting HuRI,
    Ray does it for you. And if you need more instances (e.g. more LLMs),
    you just increase num_replicas — Ray starts more containers.

HOW IT WORKS:
    OllamaService:
        - Starts an Ollama Docker container on a random free port
        - Pulls the requested model
        - Exposes generate() to send prompts and get answers
        - Scales horizontally: 2 replicas = 2 containers = 2x throughput

    QdrantService:
        - Starts a Qdrant Docker container (or reuses existing one)
        - Exposes get_url() so other services know where to connect
        - Always 1 replica: it's a database, data must be in one place
"""

import time
import socket
import subprocess

import httpx
from ray import serve


def find_free_port() -> int:
    """
    Ask the OS for a random free port.
    We need this because if we run multiple Ollama containers,
    they can't all use port 11434 — each needs its own.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def wait_for_service(url: str, timeout: int = 120) -> bool:
    """
    Returns True if ready, False if timeout.
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = httpx.get(url, timeout=5)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def is_container_running(name: str) -> bool:
    """Check if a Docker container with this name is already running."""
    result = subprocess.run(
        ["docker", "ps", "-q", "-f", f"name=^{name}$"],
        capture_output=True, text=True,
    )
    return bool(result.stdout.strip())


def remove_container(name: str):
    """Force remove a container by name (ignores errors if it doesn't exist)."""
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)


@serve.deployment
class OllamaService:
    """
    Manages one Ollama Docker container.

    LIFECYCLE:
        __init__:  starts container -> waits for it -> pulls model
        generate:  sends a prompt to the container, returns the answer
        __del__:   stops and removes the container
    """

    def __init__(
        self,
        model: str = "mistral:7b",
        image: str = "ollama/ollama:latest",
        gpu_devices: bool = False,
    ):
        self.model = model
        self.port = find_free_port()
        self.container_name = f"ollama-ray-{self.port}"
        self.base_url = f"http://localhost:{self.port}"

        remove_container(self.container_name)

        cmd = [
            "docker", "run", "-d",
            "--name", self.container_name,
            "-p", f"{self.port}:11434",
            "-v", "ollama_shared:/root/.ollama",
        ]

        if gpu_devices:
            cmd.extend([
                "--device=/dev/kfd",
                "--device=/dev/dri",
                "--group-add=video",
            ])

        cmd.append(image)

        print(f"[OllamaService] Starting container '{self.container_name}' on port {self.port}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Docker failed: {result.stderr}")

        print(f"[OllamaService] Waiting for Ollama to be ready...")
        if not wait_for_service(f"{self.base_url}/api/tags"):
            raise RuntimeError(f"Ollama didn't start within timeout on port {self.port}")

        print(f"[OllamaService] Pulling model '{model}'...")
        pull_result = subprocess.run(
            ["docker", "exec", self.container_name, "ollama", "pull", model],
            capture_output=True, text=True,
        )
        if pull_result.returncode != 0:
            raise RuntimeError(f"Failed to pull model: {pull_result.stderr}")

        print(f"[OllamaService] Ready! container='{self.container_name}', port={self.port}, model='{model}'")


    async def generate(
        self,
        messages: list,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        """
        Send messages to Ollama and return the response.
        This is what RAGHandle calls to get LLM answers.
        """
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                    },
                },
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]

    async def health(self) -> dict:
        """Check if this Ollama instance is alive."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return {"status": "ok", "port": self.port, "container": self.container_name}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def __del__(self):
        """Cleanup when Ray destroys this replica."""
        print(f"[OllamaService] Removing container '{self.container_name}'")
        remove_container(self.container_name)


@serve.deployment(num_replicas=1)
class QdrantService:
    """
    Manages a Qdrant Docker container.

    LIFECYCLE:
        __init__:  starts container (or reuses if already running)
        get_url:   returns the URL other services should connect to
        __del__:   leaves the container running (it has data!)
    """

    def __init__(
        self,
        port: int = 6333,
        image: str = "qdrant/qdrant:latest",
        storage_volume: str = "qdrant_data",
    ):
        self.port = port
        self.container_name = "qdrant-ray"
        self.url = f"http://localhost:{self.port}"

        if self._is_healthy():
            print(f"[QdrantService] Qdrant already running on port {self.port}")
            return

        remove_container(self.container_name)

        cmd = [
            "docker", "run", "-d",
            "--name", self.container_name,
            "-p", f"{self.port}:6333",
            "-v", f"{storage_volume}:/qdrant/storage",
            image,
        ]

        print(f"[QdrantService] Starting Qdrant on port {self.port}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Docker failed: {result.stderr}")

        if not wait_for_service(f"{self.url}/healthz"):
            raise RuntimeError(f"Qdrant didn't start within timeout on port {self.port}")

        print(f"[QdrantService] Ready on port {self.port}")


    def _is_healthy(self) -> bool:
        try:
            resp = httpx.get(f"{self.url}/healthz", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False


    async def get_url(self) -> str:
        """Return the URL. Called by RAGHandle to know where Qdrant is."""
        return self.url


    async def health(self) -> dict:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.url}/healthz")
                return {"status": "ok", "port": self.port, "url": self.url}
        except Exception as e:
            return {"status": "error", "error": str(e)}


    def __del__(self):
        print(f"[QdrantService] Actor destroyed. Container '{self.container_name}' left running.")


if __name__ == "__main__":
    """
    Test the services independently, without HuRI.

    Run:
        python docker_services.py

    What it does:
        1. Starts Ray
        2. Deploys QdrantService -> starts Qdrant container
        3. Deploys OllamaService -> starts Ollama container + pulls model
        4. Sends a test prompt to Ollama
        5. Prints results
        6. Ctrl+C to stop and cleanup
    """
    import ray

    print("=" * 60)
    print("Docker Services — Standalone Test")
    print("=" * 60)

    # Step 1: Start Ray
    print("\n[1/5] Starting Ray...")
    ray.init()
    serve.start()

    # Step 2: Deploy Qdrant
    print("\n[2/5] Deploying QdrantService...")
    qdrant_app = serve.run(
        QdrantService.bind(port=6333),
        name="qdrant-test",
        route_prefix="/qdrant-test",
    )
    qdrant_health = qdrant_app.health.remote().result()
    print(f"  Qdrant health: {qdrant_health}")
    qdrant_url = qdrant_app.get_url.remote().result()
    print(f"  Qdrant URL: {qdrant_url}")

    # Step 3: Deploy Ollama
    print("\n[3/5] Deploying OllamaService (this may take a minute)...")
    ollama_app = serve.run(
        OllamaService.bind(
            model="mistral:7b",
            image="ollama/ollama:rocm",  # change to ollama/ollama:latest if no AMD GPU
            gpu_devices=True,             # set False if no AMD GPU
        ),
        name="ollama-test",
        route_prefix="/ollama-test",
    )
    ollama_health = ollama_app.health.remote().result()
    print(f"  Ollama health: {ollama_health}")

    # Step 4: Test generation
    print("\n[4/5] Sending test prompt to Ollama...")
    answer = ollama_app.generate.remote(
        messages=[{"role": "user", "content": "Say hello in exactly 5 words."}],
        max_tokens=50,
    ).result()
    print(f"  Ollama response: {answer}")

    # Step 5: Done
    print("\n[5/5] All tests passed!")
    print(f"  Qdrant running at: {qdrant_url}")
    print(f"  Ollama running at: port from health check above")
    print("\nPress Ctrl+C to stop and cleanup.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        serve.shutdown()
        ray.shutdown()
        print("Done.")