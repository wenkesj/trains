from __future__ import annotations

from pygate.rules.shared_state import check


def found(module) -> list[tuple[str, int, str]]:
    return sorted((d.code, d.line, d.message.split("`")[1]) for d in check(module))


def test_a_function_writing_through_a_module_name_is_reported_at_the_write(parse):
    module = parse(
        """
        _CACHE = {}
        def remember(key, value):
            _CACHE[key] = value
        """
    )
    assert found(module) == [("PG202", 3, "_CACHE")]


def test_a_name_that_is_state_twice_over_is_reported_once_under_the_first_rule(parse):
    module = parse(
        """
        COUNT = 0
        def bump():
            global COUNT
            COUNT += 1
        """
    )
    assert found(module) == [("PG201", 3, "COUNT")]


def test_a_module_level_lock_is_reported(parse):
    assert found(parse("import threading\n_LOCK = threading.Lock()\n")) == [
        ("PG203", 2, "_LOCK")
    ]


def test_a_memoised_function_is_reported_at_its_def(parse):
    module = parse(
        """
        import functools
        @functools.lru_cache
        def load(): ...
        """
    )
    assert found(module) == [("PG204", 3, "load")]


def test_a_module_level_instance_of_a_self_writing_class_is_reported(parse):
    module = parse(
        """
        class Slot:
            def __init__(self):
                self.value = None
            def set(self, value):
                self.value = value
        SLOT = Slot()
        """
    )
    assert found(module) == [("PG205", 6, "SLOT")]


def test_constants_top_level_filling_and_caller_built_instances_are_not_state(parse):
    module = parse(
        """
        HANDLERS = {"a": print}
        TABLE = {}
        TABLE["b"] = 1
        class Record:
            def __init__(self):
                self.value = 1
        RECORD = Record()
        def make():
            local = {}
            local["x"] = 1
            return Record()
        """
    )
    assert found(module) == []
