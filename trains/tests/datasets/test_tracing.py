from trains.datasets import chat_dataset_from_tracing_provider, datum_dataset_from_chat_dataset
from trains.datum.adapters.tinker import TinkerDatumAdapter
from trains.formats.tracing import TraceRecord


def test_trace_streaming_and_datum_conversion_reopen_lazily():
    calls = []

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            return list(range(1, len(messages) + 1))

    class Spec:
        def iter_chats(self, trace, **kwargs):
            yield {
                "messages": [
                    {"role": "user", "content": "go"},
                    {"role": "assistant", "content": "north"},
                ],
                "trace_id": trace.id,
            }

    class Provider:
        reward = group_id = model_id = None
        spec = Spec()

        def iter_traces(self):
            calls.append("read")
            yield TraceRecord(id="trace", spans=[])

    chats = chat_dataset_from_tracing_provider(Provider, streaming=True)
    dataset = datum_dataset_from_chat_dataset(chats, Processor(), TinkerDatumAdapter())
    assert calls == []
    assert len(list(dataset)) == len(list(dataset)) == 1
    assert calls == ["read", "read"]
    assert set(next(iter(dataset))) == {
        "datum_json",
        "trace_id",
        "span_id",
        "group_id",
        "model_id",
        "reward",
        "metadata_json",
    }
