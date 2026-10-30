from __future__ import annotations

from pygate.rules.layout import LayoutSettings, check, definition_zone, zone_offenders


def found(module, settings: LayoutSettings | None = None) -> set[tuple[str, int]]:
    return {(d.code, d.line) for d in check(module, settings or LayoutSettings())}


def test_a_body_past_the_limit_is_reported_at_its_def_without_counting_the_docstring(parse):
    module = parse(
        '''
        def short():
            """Docstring lines are not counted."""
            a = 1
            return a

        def long():
            a = 1
            b = 2
            return a + b
        '''
    )
    assert found(module, LayoutSettings(max_function_body_lines=2)) == {("PG101", 6)}


def test_a_module_docstring_is_reported(parse):
    assert found(parse('"""A module docstring."""\nVALUE = 1\n')) == {("PG102", 1)}


def test_dunder_all_must_be_last_and_sorted(parse):
    module = parse(
        """
        __all__ = ["b", "a"]
        def a(): ...
        def b(): ...
        """
    )
    assert found(module) == {("PG103", 1), ("PG104", 1), ("PG105", 2), ("PG105", 3)}


def test_dunder_all_sorts_constants_then_classes_then_the_rest_naturally(parse):
    module = parse(
        """
        CONST = 1
        class Thing: ...
        def lane2(): ...
        def lane10(): ...
        __all__ = ["CONST", "Thing", "lane2", "lane10"]
        """
    )
    assert found(module) == set()


def test_a_public_class_after_a_private_function_is_out_of_zone(parse):
    module = parse(
        """
        def _helper(): ...
        class Public: ...
        """
    )
    assert found(module) == {("PG105", 2)}


def test_a_constant_built_from_a_definition_above_it_keeps_its_place(parse):
    assert found(parse("class Handler: ...\nHANDLERS = (Handler(),)\n")) == set()
    assert found(parse("class Handler: ...\nLIMIT = 3\n")) == {("PG105", 2)}


def test_imports_setup_and_lowercase_assignments_are_not_governed(parse):
    module = parse(
        """
        def build(): ...
        import os
        app = build()
        """
    )
    assert found(module) == set()
    assert definition_zone(module.tree.body[1]) is None
    assert definition_zone(module.tree.body[2]) is None


def test_only_an_inner_function_that_captures_nothing_is_reported(parse):
    module = parse(
        """
        def outer(x):
            def closure():
                return x
            def free():
                return 1
            @decorator
            def hook():
                return 1
            def factory():
                return 2
            return closure() + free() + hook(), factory
        """
    )
    assert found(module) == {("PG106", 4)}


def test_zone_offenders_hand_back_the_statement_a_fixer_moves(parse):
    module = parse("def _helper(): ...\nclass Public: ...\n")
    [(node, message)] = zone_offenders(module.tree)
    assert node is module.tree.body[1]
    assert "private functions" in message and "public classes" in message
