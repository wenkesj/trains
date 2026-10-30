import base64
import json

import pytest

pytest.importorskip("inspect_ai")
from inspect_ai.event import ModelEvent
from inspect_ai.log import EvalDataset, EvalLog, EvalSample, EvalSpec, write_eval_log
from inspect_ai.model import ChatMessageUser, ContentImage, GenerateConfig, ModelOutput
from inspect_ai.model._model_call import ModelCall
from inspect_ai.scorer import Score

from trains.datasets import (
    chat_dataset_from_inspect_log,
    chat_dataset_from_records,
    materialize_assets,
)
from trains.datasets.inspect import _records_from_log

MODEL = "fireworks/test-model"
IMAGE = "data:image/png;base64," + base64.b64encode(b"image-bytes").decode()


def event(model=MODEL, origin="agent"):
    return ModelEvent(
        model=model,
        origin=origin,
        input=[],
        tools=[],
        tool_choice="auto",
        config=GenerateConfig(),
        output=ModelOutput(),
        call=ModelCall(
            request={
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": IMAGE}}],
                    },
                    {"role": "assistant", "content": "previous"},
                    {"role": "user", "content": "Explain"},
                ]
            },
            response={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "reasoning_content": "look at the axes",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {"name": "zoom", "arguments": "{}"},
                                },
                            ],
                        }
                    }
                ]
            },
        ),
    )


def write_log(tmp_path, events=None, score="C"):
    log = EvalLog(
        status="success",
        eval=EvalSpec(
            created="2026-10-28T00:00:00Z",
            task="charts",
            model=MODEL,
            dataset=EvalDataset(),
            config={},
            eval_id="test-eval",
        ),
        samples=[
            EvalSample(
                id="chart-1",
                epoch=1,
                input="chart",
                target="gold",
                scores={"correctness": Score(value=score)},
                events=events or [event(), event("judge/model"), event(origin="scorer")],
            ),
        ],
    )
    path = tmp_path / "native.eval"
    write_eval_log(log, path)
    return path


def test_native_eval_retains_media_reasoning_tools_and_excludes_judges(tmp_path):
    path = write_log(tmp_path)
    records = list(_records_from_log(path))
    assert len(records) == 1
    messages = records[0]["messages"]
    assert messages[1]["weight"] == 0
    assert messages[-1]["reasoning_content"] == "look at the axes"
    assert messages[-1]["tool_calls"][0]["function"]["name"] == "zoom"
    snapshot = materialize_assets(chat_dataset_from_records(records))
    assert snapshot[0]["assets"][0]["data"] == b"image-bytes"
    assert records[0]["trace_id"] == "test-eval:chart-1:1"


def test_missing_raw_call_fails_and_unsuccessful_samples_do_not_train(tmp_path):
    missing = event().model_copy(update={"call": None})
    with pytest.raises(ValueError, match="raw model call"):
        list(_records_from_log(write_log(tmp_path, [missing])))
    with pytest.raises(ValueError, match="No accepted"):
        list(_records_from_log(write_log(tmp_path, score="I"), min_score=1))


def test_public_dataset_builder_is_replayable_and_keeps_provenance(tmp_path):
    path = write_log(tmp_path)
    dataset = chat_dataset_from_inspect_log(path, min_score=1, streaming=True)
    rows = list(dataset)
    assert rows == list(dataset)
    assert rows[0]["reward"] == 1
    assert rows[0]["group_id"] == "chart-1"
    assert rows[0]["messages"][-1]["tool_calls"][0]["function"]["name"] == "zoom"


def test_native_request_preserves_behavior_identity_and_sampling(tmp_path):
    call = event()
    call.call.request.update(
        {
            "model": "tinker://policy",
            "temperature": 1,
            "extra_body": {"top_k": -1},
            "api_key": "secret",
        }
    )
    path = write_log(tmp_path, events=[call])
    row = chat_dataset_from_inspect_log(path)[0]
    assert row["model_id"] == "tinker://policy"
    assert json.loads(row["metadata_json"])["sampling"]["top_k"] == -1
    assert "secret" not in row["metadata_json"]


def test_native_redacted_images_resolve_from_model_event_without_mutation(tmp_path):
    call = event()
    call.input = [ChatMessageUser(content=[ContentImage(image=IMAGE)])]
    call.call.request["messages"][0]["content"][0]["image_url"].update(
        url="<base64-data-removed>", detail="high"
    )
    path = write_log(tmp_path, events=[call])
    snapshot = materialize_assets(chat_dataset_from_inspect_log(path))
    assert snapshot[0]["assets"][0]["data"] == b"image-bytes"
    assert call.call.request["messages"][0]["content"][0]["image_url"]["url"] == "<base64-data-removed>"


def test_redacted_images_without_aligned_native_input_fail_clearly(tmp_path):
    call = event()
    call.call.request["messages"][0]["content"][0]["image_url"]["url"] = "<base64-data-removed>"
    with pytest.raises(ValueError, match="align redacted request images"):
        list(_records_from_log(write_log(tmp_path, events=[call])))
