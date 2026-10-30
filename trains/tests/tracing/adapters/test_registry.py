import pytest

from trains.tracing.adapters.registry import Registry, registry


def test_unknown_adapter_registration_and_format_validation():
    local = Registry()

    class CustomAdapter:
        id = "custom/v1"
        description = "custom"

        def matches(self, span):
            return True

        def extract(self, span):
            return [
                {
                    "messages": [
                        {"role": "user", "content": "go"},
                        {"role": "assistant", "content": "north"},
                    ]
                }
            ]

    adapter = CustomAdapter()
    local.register(adapter)
    assert local.convert([{}])[0][0]["messages"][-1]["content"] == "north"
    with pytest.raises(ValueError, match="already registered"):
        local.register(adapter)
    with pytest.raises(ValueError, match="Unknown trace format"):
        registry.get("missing/v1")
