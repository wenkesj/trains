from fireworks.training import sdk

from trains.datum.adapters.fireworks import FireworksDatumAdapter


def test_saved_chat_snapshot_converts_to_native_sdk_datums(native_roundtrip):
    native_roundtrip(FireworksDatumAdapter(), sdk.Datum)
