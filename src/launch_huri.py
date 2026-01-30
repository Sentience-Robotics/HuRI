import argparse
import logging
import time

import yaml

from src.core.huri import HuRI, HuriConfig
from src.modules.factory import build_module_factory


def load_config(path: str) -> HuriConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    return HuriConfig.from_dict(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="HuRI core")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to HuRI config file (YAML)",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    build_module_factory()

    huri = HuRI(config)
    time.sleep(0.1)
    try:
        huri.run()
    except KeyboardInterrupt:
        huri.stop()
    except Exception as e:
        logging.getLogger(__name__).error(e)


if __name__ == "__main__":
    main()
