import pytest
from tinker import types

from trains.datum.adapters.tinker import TinkerDatumAdapter


@pytest.mark.parametrize("bad", [[-1, 2], [True, 2], [1.5, 2]])
def test_invalid_token_ids_fail_closed(bad):
    with pytest.raises(ValueError, match="Token IDs"):
        TinkerDatumAdapter().from_tokens(bad, [3])


def test_saved_chat_snapshot_converts_to_native_sdk_datums(native_roundtrip):
    native_roundtrip(TinkerDatumAdapter(), types.Datum)
