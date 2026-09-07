# HuRI

## Presentation

HuRI is an open-source research project focused on conversational AI for humanoid robots and virtual avatars.
HuRI provides a modular architecture that allows developers to design, implement, and run AI modules within customizable conversational pipelines defined by the user.
HuRI is launched as a multi-client server, to handle multiple client (robots) conversational requests.
The framework supports the implementation and integration of multiple AI modules, including:
Speech-to-Text (STT) and Text-to-Speech (TTS), Retrieval-Augmented Generation (RAG), Emotional analysis (EMO), Motion and gesture generation (MOV)

## Getting Started

### Prerequisites

- python 3.11.14
  ```sh
  sudo apt install python3.11
  ```
- pip
  ```sh
  sudo apt install python3-pip
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/Sentience-Robotics/HuRI.git
   ```
2. Install pip packages
   ```sh
   pip install -r requirements.txt
   ```

#### Local install (single machine, no Kubernetes)

`scripts/install_local.sh` is the bare-metal counterpart of the Docker images in
`deploy/` and the Helm chart in `helm/`. It probes the machine (GPU vendor, VRAM,
RAM, disk, Python), works out which parts of the pipeline actually fit, and then
installs only those: system packages, a virtualenv with the right torch build,
the model weights, Qdrant and Ollama, plus a Ray Serve config matching the plan.

```sh
scripts/install_local.sh --plan-only   # what would run on this machine?
scripts/install_local.sh               # show the plan, then install it
```

The plan decides per module — TTS and gesture generation need an NVIDIA GPU,
STT runs fine on CPU, and the LLM tier is picked from the VRAM (or RAM) left
over. Modules that do not fit are dropped from `HURI_MODULES`, so no Serve
deployment is created for them. Override with `--force-tts`, `--force-gesture`,
`--llm-model`, `--stt-model`, … (`--help` lists everything).

It generates:

| Path | What |
| --- | --- |
| `config/huri_local.generated.yaml` | Ray Serve config for this machine |
| `config/client_local.generated.yaml` | matching client config |
| `.huri-local/huri.env` | the same runtime env vars as a plain `.env` |
| `.huri-local/secrets.env` | LLM API key, `0600`, never in the configs |
| `.huri-local/env.sh` | venv + `huri.env` + `secrets.env` in one `source` |
| `.huri-local/start.sh` / `stop.sh` / `status.sh` | run the stack |
| `.huri-local/plan.env` | the hardware plan it acted on |

```sh
.huri-local/start.sh                 # services + ray head + serve deploy
source .huri-local/env.sh
python -m src.client --config config/client_local.generated.yaml
.huri-local/status.sh                # what is up, locally and remotely
.huri-local/stop.sh --all
```

`source .huri-local/env.sh` gives a shell configured exactly like a Serve
replica (model paths, `HURI_MODULES`, TTS/gesture tuning, API key), so
`python -m src.launch_huri` and `python -m src.modules.rag.ingestion` work
without `serve deploy`.

TTS additionally needs a reference voice sample:
`--voice-sample voice.wav --voice-transcript "exactly what is said in it"`.

#### Using a remote LLM instead of a local one

Any endpoint that is not on localhost is treated as already running: it is not
installed, not started, and costs no local VRAM — which is what frees a small
GPU to run TTS and gesture generation.

```sh
# self-hosted vLLM over HTTPS with a private CA / self-signed cert
scripts/install_local.sh \
  --llm-url https://llm.example.lan --llm-provider vllm \
  --llm-model Qwen3.5-4B-GGUF --no-verify-ssl \
  --embed-url https://embedding.example.lan --embed-model bge-large-en-v1.5-gguf-Q4_K_M

# hosted OpenAI-compatible API (key goes to .huri-local/secrets.env, 0600)
scripts/install_local.sh \
  --llm-url https://api.example.com --llm-provider api \
  --llm-model some-model --llm-api-key "$MY_KEY"

# remote Qdrant too — then nothing but HuRI itself runs locally
scripts/install_local.sh --qdrant-url https://qdrant.example.lan
```

The provider selects the wire protocol (`ollama` → `/api/chat`, `vllm`/`api` →
`/v1/chat/completions`, only `api` sends the bearer token) and is inferred from
the URL and whether a key was given. Re-point an existing install without
reinstalling anything: `scripts/install_local.sh --only config --llm-url ... --llm-model ...`
then restart (`.huri-local/stop.sh && .huri-local/start.sh`).

## Usage

#### Launch HuRI server:

```sh
serve run [config_file_path]
```

We use ray serve config file, doc [here](https://docs.ray.io/en/latest/serve/configure-serve-deployment.html).

You can also launch HuRI without config file:

```sh
python -m src.launch_huri
```

#### Launch Client:

```sh
python -m src.client --config [client_config_file_path]
```

We have custom yaml file to define modules to use and how they are initialized, template [here](config/client_template.yaml).

### Folder/Module structure

- #### root

Entrypoints:

HuRI: app.py & launch_huri.py \
Client: client.py

- #### Core

HuRI's Core classes:

client & client_senders \
events \
huri \
module \
session

- #### Modules

Modules implementation:

rag: rag's implementation \
speech_to_text: speech to text implementation, including MIC (vad), STT (speech to text) and TAG (text aggregator) \
utils: utility modules, like Sender (send event to Clients)

## Developper Documentation

HuRI's complete documentation is available [here](https://docs.sentience-robotics.fr/share/p1x9ikjkhf/p/hu-ri-documentation-f4amcndYQg).
