from typing import Any

from transformers import AutoTokenizer

DEFAULT_MODEL = "LiquidAI/LFM3-Omni-2B"
LFM3_REVISION = "5797d07c125503bd4f4c0e7f915ae2d3d703e8de"


def load_tokenizer(
    model: str, *, trust_remote_code: bool = False, local_files_only: bool = False
) -> Any:
    """Load the inference tokenizer, pinning the default Liquid model revision.

    Args:
        model: Hugging Face tokenizer identity or a local snapshot directory.
        trust_remote_code: Explicitly allow custom code from a selected local snapshot.
        local_files_only: Require tokenizer files already present in the local cache.

    Returns:
        The tokenizer with the model's original chat template and special tokens.
    """
    return AutoTokenizer.from_pretrained(
        model,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code or model == DEFAULT_MODEL,
        **({"revision": LFM3_REVISION} if model == DEFAULT_MODEL else {}),
    )


def restore_liquid_generation_prefix(record: dict[str, Any]) -> dict[str, Any]:
    """Retain Liquid's forced empty thinking prefix for a captured final action.

    With thinking disabled, Liquid prefills ``<think></think>``. Its template
    omits that prefix when generated content itself contains ``</think>``.
    An explicit empty reasoning field restores the prefill without changing
    generated content or the historical messages used during inference.

    Args:
        record: One captured model call with only its final assistant supervised.

    Returns:
        The record with explicit empty reasoning when the native prefix needs it.
    """
    options = record.get("chat_template_kwargs") or {}
    messages = record.get("messages") or []
    if options.get("enable_thinking") is not False or not messages:
        return record
    final = messages[-1]
    content = final.get("content")
    if (
        final.get("role") != "assistant"
        or not final.get("weight", 1)
        or not isinstance(content, str)
        or "</think>" not in content
        or "<think>" in content
        or final.get("reasoning") is not None
        or final.get("reasoning_content") is not None
        or any(m["role"] == "assistant" and m.get("weight", 1) for m in messages[:-1])
    ):
        return record
    return {
        **record,
        "messages": [*messages[:-1], {**final, "reasoning_content": ""}],
    }
