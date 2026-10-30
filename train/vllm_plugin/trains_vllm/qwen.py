from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration as NativeQwen
from vllm.model_executor.models.utils import WeightsMapper


class Qwen3_5ForConditionalGeneration(NativeQwen):
    """Accept text-only PEFT adapter prefixes on the native multimodal model.

    Transformers' causal model saves `model.layers.*`; its multimodal counterpart
    saves `model.language_model.layers.*`. Both address the same language layers.
    Preserve native base-weight, vision and packed-projection mappings while
    adding the causal prefix, so LoRA loading cannot silently miss those layers.
    """

    hf_to_vllm_mapper = NativeQwen.hf_to_vllm_mapper | WeightsMapper(
        orig_to_new_prefix={"model.layers.": "language_model.model.layers."}
    )
