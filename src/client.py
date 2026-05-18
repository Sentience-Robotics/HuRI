import argparse
import asyncio
from typing import Dict

from omegaconf import OmegaConf

from src.core.client import Client
from src.core.dataclasses.config import ClientConfig


def load_client_config(path: str) -> ClientConfig:
    with open(path) as f:
        dict_config = OmegaConf.load(f)
        raw_resolved = OmegaConf.to_container(dict_config, resolve=True)

        if not isinstance(raw_resolved, Dict):
            raise RuntimeError("error yaml does not output a dict")

        return ClientConfig.from_dict(raw_resolved)


async def launch_client():
    parser = argparse.ArgumentParser(description="Client config")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to Client config file (YAML)",
    )

    args = parser.parse_args()
    config = load_client_config(args.config)

    await Client(config=config).run()


if __name__ == "__main__":
    asyncio.run(launch_client())
