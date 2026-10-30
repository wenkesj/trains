from baseten import loops

from trains.datum.adapters.baseten import BasetenDatumAdapter


def test_saved_chat_snapshot_converts_to_native_sdk_datums(native_roundtrip):
    native_roundtrip(BasetenDatumAdapter(), loops.Datum)
