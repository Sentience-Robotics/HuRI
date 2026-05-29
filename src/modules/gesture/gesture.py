import asyncio
import os
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

import numpy as np
from ray import serve
from ray.serve import handle

from src.core.module import Module, ModuleWithHandle
from src.modules.text_to_speech.events import Audio


_HF_REPO = os.environ.get("HURI_EMAGE_REPO", "H-Liu1997/emage_audio")
_EMAGE_SR = 16000  # EMAGE expects 16 kHz mono audio


@dataclass
class Motion:
    poses: np.ndarray        # (t, 165)  SMPL-X axis-angle, 55 joints × 3
    expressions: np.ndarray  # (t, 100)  facial expression coefficients
    trans: np.ndarray        # (t, 3)    global root translation
    fps: int = 30
    pts: float = 0.0         # presentation timestamp in seconds, paired with Audio.pts


@serve.deployment(name="GestureGeneration")
class GestureDeployment:
    def __init__(
        self,
        hf_repo: str = _HF_REPO,
        device: Optional[str] = None,
    ):
        print(f"[Gesture] importing torch...", flush=True)
        import torch
        print(f"[Gesture] importing emage...", flush=True)
        from .emage import EmageAudioModel, EmageVAEConv, EmageVQModel, EmageVQVAEConv

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"[Gesture] device={self.device} hf_repo={hf_repo!r}", flush=True)

        print("[Gesture] loading face_vq...", flush=True)
        face_vq = EmageVQVAEConv.from_pretrained(hf_repo, subfolder="emage_vq/face").to(self.device)
        print("[Gesture] loading upper_vq...", flush=True)
        upper_vq = EmageVQVAEConv.from_pretrained(hf_repo, subfolder="emage_vq/upper").to(self.device)
        print("[Gesture] loading lower_vq...", flush=True)
        lower_vq = EmageVQVAEConv.from_pretrained(hf_repo, subfolder="emage_vq/lower").to(self.device)
        print("[Gesture] loading hands_vq...", flush=True)
        hands_vq = EmageVQVAEConv.from_pretrained(hf_repo, subfolder="emage_vq/hands").to(self.device)
        print("[Gesture] loading global_ae...", flush=True)
        global_ae = EmageVAEConv.from_pretrained(hf_repo, subfolder="emage_vq/global").to(self.device)

        self.motion_vq = EmageVQModel(
            face_model=face_vq,
            upper_model=upper_vq,
            lower_model=lower_vq,
            hands_model=hands_vq,
            global_model=global_ae,
        )
        self.motion_vq.eval()

        print("[Gesture] loading EmageAudioModel...", flush=True)
        self.model = EmageAudioModel.from_pretrained(hf_repo).to(self.device)
        self.model.eval()
        print(f"[Gesture] ready", flush=True)

    def infer(self, audio_np: np.ndarray, source_sr: int = _EMAGE_SR) -> Motion:
        import torch
        import torch.nn.functional as F

        if source_sr != _EMAGE_SR:
            import librosa
            audio_np = librosa.resample(audio_np, orig_sr=source_sr, target_sr=_EMAGE_SR)

        audio_ts = torch.from_numpy(audio_np).to(self.device).unsqueeze(0)
        speaker_id = torch.zeros(1, 1, dtype=torch.long, device=self.device)

        with torch.no_grad():
            ref_trans = torch.zeros(1, 1, 3, device=self.device)
            latent_dict = self.model.inference(audio_ts, speaker_id, self.motion_vq)

            cfg = self.model.cfg
            face_latent = latent_dict["rec_face"] if cfg.lf > 0 and cfg.cf == 0 else None
            upper_latent = latent_dict["rec_upper"] if cfg.lu > 0 and cfg.cu == 0 else None
            hands_latent = latent_dict["rec_hands"] if cfg.lh > 0 and cfg.ch == 0 else None
            lower_latent = latent_dict["rec_lower"] if cfg.ll > 0 and cfg.cl == 0 else None
            face_index = torch.max(F.log_softmax(latent_dict["cls_face"], dim=2), dim=2)[1] if cfg.cf > 0 else None
            upper_index = torch.max(F.log_softmax(latent_dict["cls_upper"], dim=2), dim=2)[1] if cfg.cu > 0 else None
            hands_index = torch.max(F.log_softmax(latent_dict["cls_hands"], dim=2), dim=2)[1] if cfg.ch > 0 else None
            lower_index = torch.max(F.log_softmax(latent_dict["cls_lower"], dim=2), dim=2)[1] if cfg.cl > 0 else None

            all_pred = self.motion_vq.decode(
                face_latent=face_latent, upper_latent=upper_latent,
                lower_latent=lower_latent, hands_latent=hands_latent,
                face_index=face_index, upper_index=upper_index,
                lower_index=lower_index, hands_index=hands_index,
                get_global_motion=True, ref_trans=ref_trans[:, 0],
            )

        t = all_pred["motion_axis_angle"].shape[1]
        return Motion(
            poses=all_pred["motion_axis_angle"].cpu().numpy().reshape(t, -1),
            expressions=all_pred["expression"].cpu().numpy().reshape(t, -1),
            trans=all_pred["trans"].cpu().numpy().reshape(t, -1),
        )


class Gesture(ModuleWithHandle):
    """Gesture Module

    Consumes streaming Audio chunks produced by TTS and generates whole-body
    SMPL-X motion using the EMAGE audio-to-gesture model. Inference runs once
    per chunk so Motion events interleave with audio playback instead of all
    arriving at the end of the utterance.

    input:  audio (Audio)
    output: motion (Motion)

    :hf_repo: HuggingFace repository to load EMAGE weights from.
    :device:  PyTorch device string; defaults to CUDA when available.
    """

    _handle_cls = GestureDeployment
    input_type = "audio"
    output_type = "motion"

    def __init__(
        self,
        _handle: handle.DeploymentHandle,
    ):
        super().__init__(_handle)

    async def process(self, audio: Audio) -> AsyncGenerator[Motion, None]:  # type: ignore[override]
        if audio.data.size == 0:
            return
        motion = await self._handle.infer.remote(
            audio.data.astype(np.float32), audio.sample_rate
        )
        motion.pts = audio.pts
        yield motion
