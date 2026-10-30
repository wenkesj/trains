from trains.datum.adapters.tinker import TinkerDatumAdapter


class SpindleDatumAdapter(TinkerDatumAdapter):
    """Spindle uses native Tinker datums; deployment model capabilities still apply."""

    provider = "spindle"
