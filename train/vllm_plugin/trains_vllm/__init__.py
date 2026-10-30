from vllm import ModelRegistry


def register() -> None:
    """Register the Trains vLLM plugin entry point."""
    ModelRegistry.register_model(
        "Qwen3_5ForConditionalGeneration", "trains_vllm.qwen:Qwen3_5ForConditionalGeneration"
    )
