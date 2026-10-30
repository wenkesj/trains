import json

import tinker
from datasets import load_from_disk
from tinker import types
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from trains.datasets import chat_dataset_from_tracing_provider, datum_dataset_from_chat_dataset
from trains.datum.adapters.tinker import TinkerDatumAdapter
from trains.formats.tracing import TraceRecord, TraceSpec
from transformers import PreTrainedTokenizerFast


def test_trace_hf_disk_native_sdk_policy_training_and_resume(model_path, training_server, tmp_path):
    raw = Tokenizer(
        WordLevel({"[UNK]": 0, "user": 1, "go": 2, "assistant": 3, "north": 4}, unk_token="[UNK]")
    )
    raw.pre_tokenizer = Whitespace()
    processor = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]")
    processor.chat_template = "{% for m in messages %}{{m.role + ' ' + m.content + ' '}}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"

    class Source:
        spec = TraceSpec()
        reward = staticmethod(lambda trace: 0.8)
        group_id = staticmethod(lambda trace: "task")
        model_id = staticmethod(lambda trace: "tiny")

        def iter_traces(self):
            yield TraceRecord(
                id="rollout",
                spans=[
                    {
                        "SpanId": "model-call",
                        "SpanAttributes": {
                            "openinference.span.kind": "LLM",
                            "llm.input_messages.0.message.role": "user",
                            "llm.input_messages.0.message.content": "go",
                            "llm.output_messages.0.message.role": "assistant",
                            "llm.output_messages.0.message.content": "north",
                        },
                    }
                ],
            )

    chats = chat_dataset_from_tracing_provider(Source())
    dataset = datum_dataset_from_chat_dataset(chats, processor, TinkerDatumAdapter())
    dataset.save_to_disk(tmp_path / "dataset")
    dataset = load_from_disk(tmp_path / "dataset")
    assert dataset[0]["reward"] == 0.8
    service = tinker.ServiceClient(base_url=training_server, api_key="tml-local")
    try:
        training = service.create_lora_training_client("tiny", rank=2, train_unembed=False)
        value = json.loads(dataset[0]["datum_json"])
        datum = types.Datum(
            model_input=types.ModelInput.model_validate(value["model_input"]),
            loss_fn_inputs={
                key: types.TensorData(**tensor) for key, tensor in value["loss_fn_inputs"].items()
            },
        )
        reference = training.forward(
            [datum], "cross_entropy", loss_fn_config={"temperature": 0.7}
        ).result()
        datum.loss_fn_inputs["logprobs"] = reference.loss_fn_outputs[0]["logprobs"]
        datum.loss_fn_inputs["advantages"] = types.TensorData(
            data=[0, 0, 0.8], dtype="float32", shape=[3]
        )
        training.forward_backward([datum], "ppo", loss_fn_config={"temperature": 0.7}).result()
        assert training.optim_step(types.AdamParams()).result().metrics["step"] == 1
        state = training.save_state("iteration-1").result()
        resumed = service.create_training_client_from_state_with_optimizer(state.path)
        resumed.forward_backward([datum], "cross_entropy").result()
        assert resumed.optim_step(types.AdamParams()).result().metrics["step"] == 2
        assert resumed.save_weights_for_sampler("sampler").result().path.startswith("tinker://")
    finally:
        service.close("success").result()
