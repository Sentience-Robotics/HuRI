import argparse
import asyncio
import os
from typing import Dict

from omegaconf import OmegaConf

from src.core.client import Client
from src.core.dataclasses.config import ClientConfig

USER_ID_FILE = os.path.expanduser("~/.huri_user_id")


def load_user_id() -> str | None:
    if os.path.exists(USER_ID_FILE):
        with open(USER_ID_FILE) as f:
            return f.read().strip()
    return None


def save_user_id(_user_id: str):
    with open(USER_ID_FILE, "w") as f:
        f.write(_user_id)


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
