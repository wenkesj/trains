[Example source](https://github.com/wenkesj/trains/tree/main/examples/hf-agent-tracing) · [Session adapters](#/docs/guides/sessions)

> **Experimental:** Work in progress; convergence has not been demonstrated. Try a small run with a compatible local training service first. Hosted providers charge real money: estimate costs, verify credits, and stop unused resources.

Fine-tune `LiquidAI/LFM3-Omni-2B` on pinned
[NVIDIA SWE-Hero OpenHands trajectories](https://huggingface.co/datasets/nvidia/SWE-Hero-openhands-trajectories)
using Baseten Loops, then collect native Pi sessions for PPO.

The demonstrations are **CC BY 4.0**, attributed to NVIDIA. Source repositories
use MIT, Apache-2.0, or BSD licenses. Preparation rejects other labels and retains
attribution and repository terms in chats, datums, and `artifacts/source.json`;
preserve these notices when sharing data.

Preparation streams complete conversations without truncation. Missing tool-result
IDs are restored only when exactly one call is outstanding; ambiguous matches fail.
The trainer must support the longest conversation. Live Pi sessions use
`PiSessionAdapter`; see the [Sessions guide](#/docs/guides/sessions).

```sh
cd examples/hf-agent-tracing
git -C ../.. submodule update --init --recursive examples/tracing/exercism
uv sync --python 3.12
docker build -t trains-pi-harness:0.73.1 .
# Configure the Loops endpoint and credentials from .env.example.
uv run python run.py prepare --limit 4
uv run python run.py sft --steps 12
uv run python run.py collect --exercises anagram,binary-search,matrix --attempts 4
uv run python run.py rl --collection artifacts/collection-... --steps 1
```

Collection runs the pinned Pi CLI in a separate container for each upstream
Exercism coding attempt. Only its workspace, private profile, and native session directory
are mounted. Provider credentials stay in the host bridge; host networking lets
Pi reach it. A separate Python 3.12 container runs unchanged upstream tests with
networking disabled. Keep evaluation exercises disjoint from training, for example
`isogram,word-count`. Receipts record the Git revision and task checksum.
It writes native sessions under `~/.pi/agent/sessions/trains-hf-agent-tracing`
and preserves exact token receipts for every model call. Local sessions are
imported with `chat_dataset_from_sessions(path, adapter=PiSessionAdapter())`.
Action mismatches reject the collection before updates.

The first PPO round restores SFT weights with fresh optimizer state and uses captured sampler log probabilities
with group-centered rewards. Complete batches without sampler probabilities use
frozen teacher-forced references; metrics record the reference source. Failed attempts remain in their groups;
flat reward groups require more varied collection. The README explains checkpoints,
grading, session paths, and local tests that run Pi without allocating GPUs.
