import os
import sys
import zipfile
import tempfile
import subprocess
from pathlib import Path
from typing import List, Tuple

import cv2
import imageio
import numpy as np
import torch
import torchvision
from cog import BasePredictor, Input, Path as CogPath
from huggingface_hub import hf_hub_download, snapshot_download

# Ensure repository modules are importable
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import wan  # noqa: E402
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES  # noqa: E402

# Hugging Face sources for required assets
WAN_BASE_REPO = os.getenv("WAN_BASE_REPO", "Wan-AI/Wan2.1-T2V-14B")
WAN_ALPHA_REPO = os.getenv("WAN_ALPHA_REPO", "htdong/Wan-Alpha")
LIGHTX2V_REPO = os.getenv(
    "LIGHTX2V_REPO",
    "Kijai/WanVideo_comfy",
)

# Path where weights are cached inside the container
WAN_WEIGHTS_ROOT = Path(os.getenv("WAN_WEIGHTS_ROOT", "/weights")).resolve()

# Add 512x512 options with different base resolutions for speed/orientation
SUPPORTED_RESOLUTIONS = sorted(SUPPORTED_SIZES["t2v-14B"]) + ["512*512 (fit vertical)", "512*512 (fit horizontal)"]
DEFAULT_RESOLUTION = "832*480"
DEFAULT_NEG_PROMPT = WAN_CONFIGS["t2v-14B"].sample_neg_prompt


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def render_rgba_video(
    tensor_fgr: torch.Tensor,
    tensor_pha: torch.Tensor,
    nrow: int = 1,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Convert latent tensors into RGB preview frames and RGBA PNG-ready frames."""
    tensor_fgr = tensor_fgr.clamp(-1, 1)
    tensor_fgr = torch.stack(
        [
            torchvision.utils.make_grid(
                frame,
                nrow=nrow,
                normalize=True,
                value_range=(-1, 1),
            )
            for frame in tensor_fgr.unbind(2)
        ],
        dim=1,
    ).permute(1, 2, 3, 0)
    tensor_fgr = (tensor_fgr * 255).to(torch.uint8).cpu()

    tensor_pha = tensor_pha.clamp(-1, 1)
    tensor_pha = torch.stack(
        [
            torchvision.utils.make_grid(
                frame,
                nrow=nrow,
                normalize=True,
                value_range=(-1, 1),
            )
            for frame in tensor_pha.unbind(2)
        ],
        dim=1,
    ).permute(1, 2, 3, 0)
    tensor_pha = (tensor_pha * 255).to(torch.uint8).cpu()

    checkerboard = create_checkerboard(
        width=tensor_fgr.shape[2], height=tensor_fgr.shape[1]
    )

    composite_frames: List[np.ndarray] = []
    rgba_frames: List[np.ndarray] = []
    for frame_fgr, frame_pha in zip(tensor_fgr.numpy(), tensor_pha.numpy()):
        alpha = (
            frame_pha[:, :, 0:1]
            + frame_pha[:, :, 1:2]
            + frame_pha[:, :, 2:3]
        ) / 3.0
        rgba = np.concatenate(
            [frame_fgr[:, :, ::-1], alpha.astype(np.uint8)], axis=2
        )
        composite_frames.append(blend_checkerboard(rgba, checkerboard))
        rgba_frames.append(rgba)

    return composite_frames, rgba_frames


def create_checkerboard(
    square: int = 30, width: int = 832, height: int = 480
) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    colors = [(140, 140, 140), (113, 113, 113)]
    for y in range(0, height, square):
        for x in range(0, width, square):
            img[y : y + square, x : x + square] = colors[
                ((x // square) + (y // square)) % 2
            ]
    return img


def blend_checkerboard(rgba: np.ndarray, checkerboard: np.ndarray) -> np.ndarray:
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    fg_rgb = rgba[:, :, :3][:, :, ::-1].astype(np.float32)
    bg = cv2.resize(checkerboard, (rgba.shape[1], rgba.shape[0])).astype(np.float32)
    blended = fg_rgb * alpha + bg * (1.0 - alpha)
    return blended.astype(np.uint8)


def convert_to_format(
    rgba_frames: List[np.ndarray],
    output_path: Path,
    output_format: str,
    fps: int,
    quality: int,
    resize: str = "none",
) -> None:
    """Convert RGBA frames to webm or webp using ffmpeg."""
    workdir = output_path.parent
    temp_dir = workdir / "temp_frames"
    temp_dir.mkdir(exist_ok=True)

    # Save RGBA frames as PNGs
    for idx, frame in enumerate(rgba_frames):
        frame_path = temp_dir / f"frame_{idx:05d}.png"
        cv2.imwrite(str(frame_path), frame)

    # Build ffmpeg command
    input_pattern = str(temp_dir / "frame_%05d.png")

    if output_format == "webm":
        # WebM with VP9 codec and alpha channel
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", input_pattern,
            "-c:v", "libvpx-vp9",
            "-pix_fmt", "yuva420p",
            "-auto-alt-ref", "0",
        ]

        # Quality mapping: 100 = best (CRF 10), 1 = worst (CRF 63)
        crf = int(63 - (quality - 1) * 53 / 99)
        cmd.extend(["-crf", str(crf)])

        if resize != "none":
            width, height = resize.split("x")
            # Scale to fit within square while maintaining aspect ratio, then pad with transparency
            vf = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=0x00000000"
            cmd.extend(["-vf", vf])

        cmd.append(str(output_path))

    elif output_format == "webp":
        # Animated WebP with alpha channel
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", input_pattern,
        ]

        if resize != "none":
            width, height = resize.split("x")
            # Scale to fit within square while maintaining aspect ratio, then pad with transparency
            vf = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=0x00000000"
            cmd.extend(["-vf", vf])

        # Quality for WebP (0-100, higher is better) and loop infinitely
        # pix_fmt yuva420p ensures alpha channel is preserved
        cmd.extend([
            "-vcodec", "libwebp",
            "-pix_fmt", "yuva420p",
            "-lossless", "0",
            "-quality", str(quality),
            "-loop", "0",
        ])
        cmd.append(str(output_path))

    # Run ffmpeg
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        # Cleanup temp frames before raising error
        for f in temp_dir.glob("*.png"):
            f.unlink()
        temp_dir.rmdir()
        raise RuntimeError(f"FFmpeg failed: {e.stderr}")

    # Cleanup temp frames
    for f in temp_dir.glob("*.png"):
        f.unlink()
    temp_dir.rmdir()


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Download weights and construct the Wan-Alpha pipeline."""
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        ensure_dir(WAN_WEIGHTS_ROOT)
        ensure_dir(WAN_WEIGHTS_ROOT / "hf")

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        checkpoint_dir = self._download_wan_base()
        self.vae_lora_path = self._download_file(
            WAN_ALPHA_REPO, "decoder.bin", "wan_alpha"
        )
        self.dora_path = self._download_file(
            WAN_ALPHA_REPO, "epoch-13-1500.safetensors", "wan_alpha"
        )
        self.lightx2v_path = self._download_file(
            LIGHTX2V_REPO,
            "Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors",
            "lightx2v",
        )

        self.pipeline = wan.WanT2V_dora_lightx2v(
            config=WAN_CONFIGS["t2v-14B"],
            checkpoint_dir=str(checkpoint_dir),
            vae_lora_checkpoint=str(self.vae_lora_path),
            lora_path=str(self.dora_path),
            lightx2v_path=str(self.lightx2v_path),
            lora_ratio=1.0,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
            t5_cpu=False,
        )

    def predict(
        self,
        prompt: str = Input(description="Text prompt for video generation."),
        negative_prompt: str = Input(
            default="",
            description="Optional negative prompt. If blank, uses the model default.",
        ),
        resolution: str = Input(
            default=DEFAULT_RESOLUTION,
            choices=SUPPORTED_RESOLUTIONS,
            description="Video resolution formatted as width*height.",
        ),
        num_frames: int = Input(
            default=81,
            description="Total frames (must satisfy 4n+1; default ≈5s at 16fps).",
        ),
        sample_steps: int = Input(
            default=4,
            description="Number of diffusion sampling steps.",
        ),
        guidance_scale: float = Input(
            default=1.0,
            description="Classifier-free guidance scale.",
        ),
        solver: str = Input(
            default="unipc",
            choices=["unipc", "dpm++"],
            description="Sampling solver.",
        ),
        seed: int = Input(
            default=-1,
            description="Random seed (-1 for random).",
        ),
        output_format: str = Input(
            default="webm",
            choices=["webm", "webp"],
            description="Output video format with transparency support.",
        ),
        fps: int = Input(
            default=16,
            ge=1,
            le=60,
            description="Output framerate (1-60 fps).",
        ),
        output_quality: int = Input(
            default=85,
            ge=1,
            le=100,
            description="Output quality (1-100, higher is better).",
        ),
    ) -> CogPath:
        if num_frames % 4 != 1:
            raise ValueError("`num_frames` must satisfy 4n + 1 (e.g. 33, 49, 81).")

        # Handle 512x512 workarounds: render at supported resolution then fit with transparent padding
        actual_resolution = resolution
        target_resize = "none"

        if resolution == "512*512 (fit vertical)":
            actual_resolution = "480*832"  # Render portrait (faster than 720*1280)
            target_resize = "512x512"  # Fit into 512x512 with transparent padding
        elif resolution == "512*512 (fit horizontal)":
            actual_resolution = "832*480"  # Render landscape (faster than 1280*720)
            target_resize = "512x512"  # Fit into 512x512 with transparent padding

        size = SIZE_CONFIGS[actual_resolution]
        neg_prompt = negative_prompt or DEFAULT_NEG_PROMPT

        videos_fgr, videos_pha = self.pipeline.generate(
            input_prompt=prompt,
            size=size,
            frame_num=num_frames,
            shift=5.0,
            sample_solver=solver,
            sampling_steps=sample_steps,
            guide_scale=guidance_scale,
            n_prompt=neg_prompt,
            seed=seed,
            offload_model=True,
        )

        composite_frames, rgba_frames = render_rgba_video(
            videos_fgr.unsqueeze(0), videos_pha.unsqueeze(0)
        )

        workdir = Path(tempfile.mkdtemp())

        # Generate output file based on format
        output_ext = "webm" if output_format == "webm" else "webp"
        output_path = workdir / f"output.{output_ext}"

        # Convert to requested format using ffmpeg
        convert_to_format(
            rgba_frames=rgba_frames,
            output_path=output_path,
            output_format=output_format,
            fps=fps,
            quality=output_quality,
            resize=target_resize,
        )

        return CogPath(str(output_path))

    def _download_wan_base(self) -> Path:
        target_dir = ensure_dir(WAN_WEIGHTS_ROOT / "wan2.1_t2v_14b")
        if not any(target_dir.iterdir()):
            snapshot_download(
                repo_id=WAN_BASE_REPO,
                local_dir=target_dir,
                local_dir_use_symlinks=False,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.bin",
                    "*.pt",
                    "*.pth",
                    "*.model",
                    "*.txt",
                    "*.py",
                ],
            )
        return target_dir

    def _download_file(self, repo_id: str, filename: str, subdir: str) -> Path:
        target_dir = ensure_dir(WAN_WEIGHTS_ROOT / subdir)
        return Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=target_dir,
                local_dir_use_symlinks=False,
            )
        )
