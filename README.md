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
