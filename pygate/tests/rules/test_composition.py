from __future__ import annotations

from pygate.rules.composition import CompositionSettings, check


def found(module, settings: CompositionSettings | None = None) -> list[tuple[str, int]]:
    return sorted((d.code, d.line) for d in check(module, settings or CompositionSettings()))


def class_with_slots(count: int) -> str:
    slots = "\n".join(f"        self.slot{index} = {index}" for index in range(count))
    return f"class Holder:\n    def __init__(self):\n{slots}\n"


def class_with_methods(count: int) -> str:
    return "class Holder:\n" + "".join(f"    def m{i}(self): ...\n" for i in range(count))


def test_slots_past_the_limit_are_reported_at_the_class(parse):
    assert found(parse(class_with_slots(12))) == []
    assert found(parse(class_with_slots(13))) == [("PG301", 1)]


def test_methods_past_the_limit_are_reported_at_the_class(parse):
    assert found(parse(class_with_methods(12))) == []
    assert found(parse(class_with_methods(13))) == [("PG302", 1)]


def test_declared_dataclass_fields_are_not_slots(parse):
    fields = "".join(f"    field{index}: int = {index}\n" for index in range(20))
    assert found(parse(f"@dataclass\nclass Record:\n{fields}")) == []


def test_the_thresholds_come_from_settings(parse):
    settings = CompositionSettings(max_instance_attributes=2, max_methods=1)
    assert found(parse(class_with_slots(3)), settings) == [("PG301", 1)]
