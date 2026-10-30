# Native Liquid prompt rendering for the token-sampler bridge.

import copy
import json

from baseten import loops
from recipes.models import DEFAULT_MODEL, load_tokenizer
from tinker_cookbook import model_info
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.qwen3_5 import Qwen3_5Renderer

LIQUID_OPTIONS = {"enable_thinking": True, "preserve_thinking": True}


class LiquidRenderer:
    """Use Liquid's HF template and the upstream shared XML tool-call decoder."""

    def __init__(self, tokenizer):
        """Bind the exact Liquid tokenizer used for preparation and sampling."""
        self.tokenizer = tokenizer
        # Liquid and Qwen3.5 share the XML tool-call wire format. Only reuse the
        # upstream response decoder; every input token comes from Liquid's template.
        self._decoder = _LiquidDecoder(tokenizer, strip_thinking_from_history=False)

    def build_generation_prompt(self, messages, *, tools=None):
        """Render the complete wire conversation with native Liquid tool schemas."""
        messages = copy.deepcopy(messages)
        for message in messages:
            for call in message.get("tool_calls") or []:
                function = call["function"]
                if isinstance(function.get("arguments"), str):
                    function["arguments"] = json.loads(function["arguments"])
        tokens = self.tokenizer.apply_chat_template(
            messages,
            tools=tools or None,
            **LIQUID_OPTIONS,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
        )
        return loops.ModelInput(chunks=[loops.EncodedTextChunk(tokens=tokens)])

    def get_stop_sequences(self):
        """Return Liquid's native end-of-message token."""
        return [self.tokenizer.convert_tokens_to_ids("<|im_end|>")]

    def parse_response(self, tokens):
        """Decode generated reasoning and XML calls while retaining raw receipts."""
        prefix = self.tokenizer.encode("<think>", add_special_tokens=False)
        message, termination = self._decoder.parse_response([*prefix, *tokens])
        content = message.get("content")
        if not termination.is_clean and isinstance(content, str):
            # An unfinished response is returned as raw text by the upstream
            # decoder. Remove only our prefilled token, never generated text.
            message["content"] = content.removeprefix("<think>")
        return message, termination


class _LiquidDecoder(Qwen3_5Renderer):
    """Reuse XML decoding without Qwen's model-specific thinking prefill."""

    def _normalize_response_tokens(self, response: list[int]) -> list[int]:
        """Keep the Liquid prefill supplied by the bridge exactly once."""
        return response


def load_renderer(model=DEFAULT_MODEL, tokenizer=None):
    """Select native Liquid rendering or a model's registered Cookbook renderer."""
    tokenizer = tokenizer if tokenizer is not None else load_tokenizer(model)
    if model.startswith("LiquidAI/LFM3-"):
        return LiquidRenderer(tokenizer)
    return get_renderer(
        model_info.get_recommended_renderer_name(model), tokenizer, model_name=model
    )
