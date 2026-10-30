from __future__ import annotations

from pygate.rules.cohesion import CohesionSettings, check


def found(module, settings: CohesionSettings | None = None) -> dict[str, str]:
    return {d.code: d.message for d in check(module, settings or CohesionSettings()) if d.line == 1}


def test_two_disconnected_clusters_are_two_concepts(parse):
    module = parse(
        """
        def a(): return b()
        def b(): return 1
        def c(): return d()
        def d(): return 2
        """
    )
    report = found(module)
    assert set(report) == {"PG401"}
    assert "[a, b]" in report["PG401"] and "[c, d]" in report["PG401"]


def test_a_shared_module_symbol_joins_clusters(parse):
    module = parse(
        """
        LIMIT = 1
        def a(): return b() + LIMIT
        def b(): return 1
        def c(): return d() + LIMIT
        def d(): return 2
        """
    )
    assert found(module) == {}


def test_lone_helpers_are_not_rival_concepts(parse):
    assert found(parse("def a(): ...\ndef b(): ...\n")) == {}


def test_the_public_surface_is_dunder_all_when_declared(parse):
    functions = "".join(f"def f{index}(): ...\n" for index in range(13))
    assert set(found(parse(functions))) == {"PG402"}
    assert found(parse(functions + '__all__ = ["f0", "f1"]\n')) == {}


def test_width_is_measured_against_its_own_threshold(parse):
    functions = "".join(f"def f{index}(): ...\n" for index in range(6))
    assert set(found(parse(functions), CohesionSettings(max_top_level_definitions=5))) == {"PG403"}
