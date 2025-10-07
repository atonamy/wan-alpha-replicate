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
    print(f"[render_rgba_video] Processing tensors - fgr shape: {tensor_fgr.shape}, pha shape: {tensor_pha.shape}")
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
    print(f"[render_rgba_video] Using np.maximum() for alpha channel extraction")
    for frame_idx, (frame_fgr, frame_pha) in enumerate(zip(tensor_fgr.numpy(), tensor_pha.numpy())):
        if frame_idx == 0:
            # Log raw alpha channel values BEFORE processing
            print(f"[render_rgba_video] Frame 0 RAW alpha channels - R: [{frame_pha[:,:,0].min():.2f}, {frame_pha[:,:,0].max():.2f}], G: [{frame_pha[:,:,1].min():.2f}, {frame_pha[:,:,1].max():.2f}], B: [{frame_pha[:,:,2].min():.2f}, {frame_pha[:,:,2].max():.2f}]")

        # Use max instead of average to preserve strongest alpha signal
        alpha = np.maximum(
            np.maximum(frame_pha[:, :, 0:1], frame_pha[:, :, 1:2]),
            frame_pha[:, :, 2:3]
        )

        if frame_idx == 0:
            # Log AFTER np.maximum processing
            print(f"[render_rgba_video] Frame 0 AFTER np.maximum() - alpha range: [{alpha.min():.2f}, {alpha.max():.2f}], mean: {alpha.mean():.2f}")
            # Log sample of actual pixel values in center of frame
            h, w = alpha.shape[:2]
            center_alpha = alpha[h//2-10:h//2+10, w//2-10:w//2+10]
            print(f"[render_rgba_video] Frame 0 center region (20x20px) alpha - min: {center_alpha.min():.2f}, max: {center_alpha.max():.2f}, mean: {center_alpha.mean():.2f}")

        # Convert to uint8 (0-255 range)
        alpha_uint8 = alpha.astype(np.uint8)

        if frame_idx == 0:
            print(f"[render_rgba_video] Frame 0 AFTER uint8 conversion - alpha range: [{alpha_uint8.min()}, {alpha_uint8.max()}], mean: {alpha_uint8.mean():.2f}")

        rgba = np.concatenate(
            [frame_fgr[:, :, ::-1], alpha_uint8], axis=2
        )

        if frame_idx == 0:
            print(f"[render_rgba_video] Frame 0 FINAL RGBA - shape: {rgba.shape}, alpha channel: [{rgba[:,:,3].min()}, {rgba[:,:,3].max()}], mean: {rgba[:,:,3].mean():.2f}")

        composite_frames.append(blend_checkerboard(rgba, checkerboard))
        rgba_frames.append(rgba)

    print(f"[render_rgba_video] Generated {len(rgba_frames)} RGBA frames")
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
    input_fps: float,
    output_fps: int,
    quality: int,
    resize: str = "none",
    remove_first_frame: bool = False,
    remove_last_frame: bool = False,
) -> None:
    """Convert RGBA frames to webm or webp using ffmpeg.

    Args:
        input_fps: The framerate at which frames were generated (real_fps)
        output_fps: The desired output framerate (user-specified fps)
        remove_first_frame: Remove first frame from final video
        remove_last_frame: Remove last frame from final video
    """
    workdir = output_path.parent
    temp_dir = workdir / "temp_frames"
    temp_dir.mkdir(exist_ok=True)

    # Save RGBA frames as PNGs
    print(f"[convert_to_format] Saving {len(rgba_frames)} RGBA frames as PNGs...")
    for idx, frame in enumerate(rgba_frames):
        if idx == 0:
            print(f"[convert_to_format] Frame 0 before PNG save - RGBA shape: {frame.shape}, alpha: [{frame[:,:,3].min()}, {frame[:,:,3].max()}], mean: {frame[:,:,3].mean():.2f}")
        frame_path = temp_dir / f"frame_{idx:05d}.png"
        cv2.imwrite(str(frame_path), frame)

    # Build ffmpeg command
    input_pattern = str(temp_dir / "frame_%05d.png")

    # Build video filter chain
    vf_filters = []

    # IMPORTANT: fps filter must come FIRST to do framerate conversion
    # Then select filter operates on the converted frames (final video frames)
    vf_filters.append(f"fps={output_fps}")

    # Add frame trimming filter if requested (operates on frames AFTER fps conversion)
    if remove_first_frame and remove_last_frame:
        # Calculate expected output frames after fps conversion
        total_frames = int((len(rgba_frames) / input_fps) * output_fps)
        vf_filters.append(f"select='between(n\\,1\\,{total_frames-2})'")
        vf_filters.append("setpts=N/FRAME_RATE/TB")
    elif remove_first_frame:
        vf_filters.append("select='gte(n\\,1)'")
        vf_filters.append("setpts=N/FRAME_RATE/TB")
    elif remove_last_frame:
        # Calculate expected output frames after fps conversion
        total_frames = int((len(rgba_frames) / input_fps) * output_fps)
        vf_filters.append(f"select='lt(n\\,{total_frames-1})'")
        vf_filters.append("setpts=N/FRAME_RATE/TB")

    # Add resize filter if requested (applied last)
    if resize != "none":
        width, height = resize.split("x")
        vf_filters.append(f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=0x00000000")

    if output_format == "webm":
        # WebM with VP9 codec and alpha channel
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(input_fps),  # Input framerate (how frames were generated)
            "-i", input_pattern,
            "-c:v", "libvpx-vp9",
            "-pix_fmt", "yuva420p",
            "-auto-alt-ref", "0",
        ]

        # Quality mapping: 100 = best (CRF 10), 1 = worst (CRF 63)
        crf = int(63 - (quality - 1) * 53 / 99)
        cmd.extend(["-crf", str(crf)])

        # Apply filter chain (fps conversion + frame removal + resize)
        cmd.extend(["-vf", ",".join(vf_filters)])

        cmd.append(str(output_path))

    elif output_format == "webp":
        # Animated WebP with alpha channel
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(input_fps),  # Input framerate (how frames were generated)
            "-i", input_pattern,
        ]

        # Apply filter chain (fps conversion + frame removal + resize)
        cmd.extend(["-vf", ",".join(vf_filters)])

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
    print(f"[convert_to_format] Running FFmpeg: {' '.join(cmd[:10])}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[convert_to_format] FFmpeg encoding complete - output: {output_path.name}")
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
        print("="*80)
        print("🔥 SETUP STARTING - VERSION 1.0.1 WITH ENHANCED ALPHA LOGGING 🔥")
        print("="*80)
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
            ge=5,
            le=121,
            description="Desired number of frames (5-121). Internally adjusted to valid 4n+1 values (min 81, max 121).",
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
            ge=5,
            le=30,
            description="Output framerate (5-30 fps).",
        ),
        output_quality: int = Input(
            default=85,
            ge=1,
            le=100,
            description="Output quality (1-100, higher is better).",
        ),
        remove_first_frame: bool = Input(
            default=False,
            description="Remove the first frame from the final output video.",
        ),
        remove_last_frame: bool = Input(
            default=False,
            description="Remove the last frame from the final output video.",
        ),
    ) -> CogPath:
        print(f"[predict] Starting generation - prompt: '{prompt[:50]}...', resolution: {resolution}")

        # Smart frame adjustment: map desired frames to valid 4n+1 values (min 81, max 121)
        VALID_FRAMES = [81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121]

        # Calculate desired duration
        desired_duration = num_frames / fps

        # Determine actual frames to generate
        if num_frames < 81:
            actual_frames = 81  # Use minimum
        else:
            # Find nearest valid frame count >= num_frames
            actual_frames = next((f for f in VALID_FRAMES if f >= num_frames), 121)

        # Calculate real fps for generation (to match desired duration)
        real_fps = actual_frames / desired_duration

        print(f"[predict] Frame calculation - requested: {num_frames}, actual: {actual_frames}, real_fps: {real_fps:.2f}, output_fps: {fps}")

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

        print(f"[predict] Generation params - size: {size}, frames: {actual_frames}, steps: {sample_steps}, guidance: {guidance_scale}, solver: {solver}, seed: {seed}")
        print(f"[predict] Calling pipeline.generate()...")

        videos_fgr, videos_pha = self.pipeline.generate(
            input_prompt=prompt,
            size=size,
            frame_num=actual_frames,
            shift=5.0,
            sample_solver=solver,
            sampling_steps=sample_steps,
            guide_scale=guidance_scale,
            n_prompt=neg_prompt,
            seed=seed,
            offload_model=True,
        )

        print(f"[predict] Pipeline generation complete. Processing output tensors...")

        composite_frames, rgba_frames = render_rgba_video(
            videos_fgr.unsqueeze(0), videos_pha.unsqueeze(0)
        )

        workdir = Path(tempfile.mkdtemp())

        # Generate output file based on format
        output_ext = "webm" if output_format == "webm" else "webp"
        output_path = workdir / f"output.{output_ext}"

        print(f"[predict] Converting to {output_format} format - quality: {output_quality}, resize: {target_resize}")

        # Convert to requested format using ffmpeg
        convert_to_format(
            rgba_frames=rgba_frames,
            output_path=output_path,
            output_format=output_format,
            input_fps=real_fps,
            output_fps=fps,
            quality=output_quality,
            resize=target_resize,
            remove_first_frame=remove_first_frame,
            remove_last_frame=remove_last_frame,
        )

        print(f"[predict] Video generation complete - output: {output_path}")
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
