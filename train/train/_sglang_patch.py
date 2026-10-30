# Apply pinned LFM3 LoRA compatibility fixes before starting SGLang.

import os
import subprocess
import sys
from pathlib import Path


def _apply_patch(source: Path, patch: Path) -> None:
    """Apply the complete patch once, rejecting incompatible or partial sources."""
    command = [
        "patch",
        "--batch",
        "--silent",
        "--strip=1",
        "--directory",
        str(source),
        "--input",
        str(patch.resolve()),
    ]
    forward = subprocess.run([*command, "--forward", "--dry-run"], capture_output=True, text=True)
    if forward.returncode == 0:
        subprocess.run([*command, "--forward"], check=True)
        return
    reverse = subprocess.run(
        [*command, "--reverse", "--force", "--dry-run"], capture_output=True, text=True
    )
    if reverse.returncode != 0:
        raise RuntimeError(
            "LFM3 LoRA patch does not match this SGLang source. Use the pinned Compose image.\n"
            + forward.stdout
            + forward.stderr
        )


def _main() -> None:
    """Patch the container source and replace this process with the native server."""
    _apply_patch(Path("/sgl-workspace/sglang"), Path(__file__).with_name("sglang-lfm3-lora.patch"))
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        *sys.argv[1:],
        "--enable-lora",
        "--lora-backend=triton",
        "--max-lora-rank=" + os.environ.get("INFERENCE_MAX_LORA_RANK", "8"),
        "--max-loras-per-batch=2",
        "--lora-target-modules",
        "qkv_proj",
        "o_proj",
        "out_proj",
        "in_proj",
        "gate_up_proj",
        "down_proj",
        "self_attn.gate_proj",
    ]
    if os.environ.get("INFERENCE_DISABLE_CUDA_GRAPH", "false").lower() in {"1", "true"}:
        command.append("--disable-cuda-graph")
    os.execv(sys.executable, command)


if __name__ == "__main__":
    _main()
