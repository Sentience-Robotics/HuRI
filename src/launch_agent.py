import argparse
import logging
import time

import yaml

from src.core.agent import Agent, AgentConfig, HuriConfig


def load_config(path: str) -> AgentConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    return AgentConfig.from_dict(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="HuRI core")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to HuRI config file (YAML)",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    agent = Agent(config)
    time.sleep(0.1)
    try:
        agent.run()
    except KeyboardInterrupt:
        agent.stop_all()
    except Exception as e:
        logging.getLogger(__name__).error(e)


if __name__ == "__main__":
    main()
