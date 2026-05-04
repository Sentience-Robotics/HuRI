# HuRI

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
