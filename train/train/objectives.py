import torch

from train.models import Batch, Datum


def validate_batch(batch: Batch, vocab_size: int, context: int) -> None:
    """Validate the entire batch before any gradients can be accumulated."""
    config = batch.loss_fn_config or {}
    allowed = {"temperature", "skip_token_normalization"} | (
        {"clip_low_threshold", "clip_high_threshold"} if batch.loss_fn == "ppo" else set()
    )
    if config.keys() - allowed:
        raise ValueError("Unsupported loss configuration")
    if config.get("skip_token_normalization", 0) not in {0, 1}:
        raise ValueError("skip_token_normalization must be boolean")
    if batch.loss_fn == "ppo":
        low = float(config.get("clip_low_threshold", 0.8))
        high = float(config.get("clip_high_threshold", 1.2))
        if not 0 < low <= 1 <= high:
            raise ValueError("Invalid PPO clipping thresholds")
    if float(config.get("temperature", 1)) <= 0:
        raise ValueError("Temperature must be positive")
    for datum in batch.data:
        tokens = datum.model_input.tokens()
        if not tokens or len(tokens) > context or any(t < 0 or t >= vocab_size for t in tokens):
            raise ValueError("Input tokens are empty, out of vocabulary, or exceed model context")
        targets = datum.loss_fn_inputs.get("target_tokens")
        if (
            targets is None
            or targets.dtype != "int64"
            or (targets.shape or [len(targets.data)]) != [len(tokens)]
        ):
            raise ValueError(
                "target_tokens must be a one-dimensional int64 tensor matching input length"
            )
        if any(t < 0 or t >= vocab_size for t in targets.data):
            raise ValueError("Target token is out of vocabulary")
        required = {"logprobs", "advantages"} if batch.loss_fn != "cross_entropy" else set()
        if not required <= datum.loss_fn_inputs.keys():
            raise ValueError("RL losses require logprobs and advantages")
        if datum.loss_fn_inputs.keys() - {"target_tokens", "weights", "logprobs", "advantages"}:
            raise ValueError("Unsupported loss input")
        if any(
            (value.shape or [len(value.data)]) != (targets.shape or [len(targets.data)])
            for value in datum.loss_fn_inputs.values()
        ):
            raise ValueError("Loss inputs must match input length")


def objective(logprobs: torch.Tensor, datum: Datum, batch: Batch) -> torch.Tensor:
    """Compute the requested native loss from token scores and masks."""
    weights = (
        torch.tensor(datum.loss_fn_inputs["weights"].data, device=logprobs.device)
        if "weights" in datum.loss_fn_inputs
        else torch.ones_like(logprobs)
    )
    weights = weights.reshape(logprobs.shape)
    if batch.loss_fn == "cross_entropy":
        return -(logprobs * weights).sum()
    old = torch.tensor(datum.loss_fn_inputs["logprobs"].data, device=logprobs.device)
    advantages = torch.tensor(datum.loss_fn_inputs["advantages"].data, device=logprobs.device)
    ratio = (logprobs - old).exp()
    if batch.loss_fn == "ppo":
        config = batch.loss_fn_config or {}
        low = float(config.get("clip_low_threshold", 0.8))
        high = float(config.get("clip_high_threshold", 1.2))
        if not 0 < low <= 1 <= high:
            raise ValueError("Invalid PPO clipping thresholds")
        return -(
            torch.minimum(ratio * advantages, ratio.clamp(low, high) * advantages) * weights
        ).sum()
    return -(ratio * advantages * weights).sum()
