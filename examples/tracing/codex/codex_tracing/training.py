# Native Tinker checkpoint publication and one behavior-policy PPO update.

import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import tinker
from datasets import Dataset
from pydantic import BaseModel, Field
from recipes.models import DEFAULT_MODEL, load_tokenizer, restore_liquid_generation_prefix
from recipes.rl import group_advantages, ppo_update, validate_sampling
from tinker import types
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer
from trains.datasets import datum_dataset_from_chat_dataset
from trains.datum.adapters.cookbook import CookbookRenderer
from trains.datum.adapters.tinker import TinkerDatumAdapter


class Policy(BaseModel):
    """A published sampler and its matching optimizer state and rendering settings."""

    model: str
    state: str
    base_model: str
    training_url: str
    inference_url: str
    tokenizer_model: str
    renderer: str | None = None
    rendering_options: dict = Field(default_factory=dict)

    def save(self, destination: Path) -> None:
        """Write the policy receipt as JSON, creating its parent directory."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.model_dump_json(indent=2) + "\n")


class ConfiguredRenderer(CookbookRenderer):
    """Require recorded HF options to match the explicitly configured native renderer."""

    def __init__(self, renderer: Any, options: dict) -> None:
        """Require receipt rendering settings to match the native renderer.

        Args:
            renderer: Native renderer responsible for exact chat and media token layout.
            options: Chat-template settings recorded in the policy receipt.
        """
        super().__init__(renderer)
        if options and not (renderer.disables_thinking and options == {"enable_thinking": False}):
            raise ValueError("Native renderer does not implement the selected rendering options")
        self.options = options

    def build_supervised_examples(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> list:
        """Render complete messages while preserving the configured assistant masks."""
        if (chat_template_kwargs or {}) != self.options:
            raise ValueError("Captured rendering settings differ from the publication receipt")
        return super().build_supervised_examples(messages, tools=tools)


def initial_policy(
    *,
    base_model: str,
    training_url: str,
    inference_url: str,
    tokenizer_model: str,
    renderer: str | None = None,
    rendering_options: dict | None = None,
    rank: int = 8,
) -> Policy:
    """Create LoRA state, save optimizer weights, and publish a native sampler."""
    service = tinker.ServiceClient(base_url=training_url, api_key=os.environ["TINKER_API_KEY"])
    try:
        client = service.create_lora_training_client(
            base_model=base_model, rank=rank, train_unembed=False
        )
        name = "tracing-initial-" + uuid4().hex
        state = client.save_state(name).result()
        sampler = client.save_weights_for_sampler(name).result()
        service.create_sampling_client(model_path=sampler.path)
        return Policy(
            model=sampler.path,
            state=state.path,
            base_model=base_model,
            training_url=training_url,
            inference_url=inference_url.rstrip("/"),
            tokenizer_model=tokenizer_model,
            renderer=renderer,
            rendering_options=rendering_options or {},
        )
    finally:
        service.close("success").result()


def train_once(
    chats: Dataset, policy: Policy, *, learning_rate: float = 1e-5, steps: int = 1
) -> tuple[Policy, dict]:
    """Restore collection state, apply PPO updates, and publish the new policy."""
    rows = list(chats)
    group_advantages(rows, behavior_model=policy.model)
    validate_sampling(rows)
    processor = (
        ConfiguredRenderer(
            get_renderer(
                policy.renderer,
                get_tokenizer(policy.tokenizer_model),
                model_name=policy.tokenizer_model,
            ),
            policy.rendering_options,
        )
        if policy.renderer
        else load_tokenizer(
            policy.tokenizer_model, trust_remote_code=policy.base_model == DEFAULT_MODEL
        )
    )
    if policy.base_model == DEFAULT_MODEL and not policy.renderer:
        chats = chats.map(restore_liquid_generation_prefix)
    datums = datum_dataset_from_chat_dataset(chats, processor, TinkerDatumAdapter())
    service = tinker.ServiceClient(
        base_url=policy.training_url, api_key=os.environ["TINKER_API_KEY"]
    )
    try:
        client = service.create_training_client_from_state_with_optimizer(policy.state)
        metrics = ppo_update(
            list(datums),
            client,
            types,
            behavior_model=policy.model,
            steps=steps,
            learning_rate=learning_rate,
        )
        name = "tracing-update-" + uuid4().hex
        state = client.save_state(name).result()
        sampler = client.save_weights_for_sampler(name).result()
        service.create_sampling_client(model_path=sampler.path)
        return policy.model_copy(update={"model": sampler.path, "state": state.path}), metrics
    finally:
        service.close("success").result()
