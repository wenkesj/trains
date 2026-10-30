import unittest

from trains_vllm.qwen import Qwen3_5ForConditionalGeneration
from vllm.lora.utils import parse_fine_tuned_lora_name
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration as NativeQwen


class MappingTests(unittest.TestCase):
    def test_text_and_multimodal_loras_address_same_language_layers(self):
        mapper = Qwen3_5ForConditionalGeneration.hf_to_vllm_mapper
        for module in ("mlp.down_proj", "linear_attn.in_proj_qkv", "self_attn.q_proj"):
            for suffix in ("lora_A", "lora_B"):
                causal = f"base_model.model.model.layers.0.{module}.{suffix}.weight"
                multimodal = causal.replace("model.layers.", "model.language_model.layers.")
                actual = parse_fine_tuned_lora_name(causal, mapper)
                self.assertEqual(actual, parse_fine_tuned_lora_name(multimodal, mapper))
                self.assertEqual(actual[0], "language_model.model.layers.0." + module)

    def test_native_base_and_vision_mappings_stay_identical(self):
        parent = NativeQwen.hf_to_vllm_mapper
        mapper = Qwen3_5ForConditionalGeneration.hf_to_vllm_mapper
        for key in (
            "model.language_model.layers.0.mlp.down_proj.weight",
            "model.visual.blocks.0.attn.qkv.weight",
            "lm_head.weight",
        ):
            self.assertEqual(mapper._map_name(key), parent._map_name(key))
        self.assertEqual(
            parent._map_name("model.layers.0.mlp.down_proj.weight"),
            "model.layers.0.mlp.down_proj.weight",
        )
        self.assertIs(
            Qwen3_5ForConditionalGeneration._processor_factory, NativeQwen._processor_factory
        )
