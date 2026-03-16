import time

import ray

from src.core.huri import Dict, HuRI, handle, serve
from src.modules.speech_to_text.speech_to_text import STTHandle


def main() -> None:
    ray.init()

    services: Dict[str, handle.DeploymentHandle] = {
        "stt": STTHandle.bind(),
    }
    app = HuRI.bind("", services)
    time.sleep(0.1)
    try:
        serve.run(app, name="HuRI", blocking=True)
    except KeyboardInterrupt:
        return
    except Exception as e:
        ray.logger.error(e)


if __name__ == "__main__":
    main()
