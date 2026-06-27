from omegaconf import OmegaConf
from transformers import PretrainedConfig


class EmageAudioConfig(PretrainedConfig):
    model_type = "emage_audio"

    def __init__(self, config_obj=None, **kwargs):
        if config_obj is not None:
            cfg_dict = OmegaConf.to_container(config_obj, resolve=True)
            if isinstance(cfg_dict, dict):
                kwargs.update({str(k): v for k, v in cfg_dict.items()})
        super().__init__(**kwargs)


class EmageVQVAEConvConfig(PretrainedConfig):
    model_type = "emage_vqvaeconv"

    def __init__(self, config_obj=None, **kwargs):
        if config_obj is not None:
            cfg_dict = OmegaConf.to_container(config_obj, resolve=True)
            if isinstance(cfg_dict, dict):
                kwargs.update({str(k): v for k, v in cfg_dict.items()})
        super().__init__(**kwargs)


class EmageVAEConvConfig(PretrainedConfig):
    model_type = "emage_vaeconv"

    def __init__(self, config_obj=None, **kwargs):
        if config_obj is not None:
            cfg_dict = OmegaConf.to_container(config_obj, resolve=True)
            if isinstance(cfg_dict, dict):
                kwargs.update({str(k): v for k, v in cfg_dict.items()})
        super().__init__(**kwargs)
