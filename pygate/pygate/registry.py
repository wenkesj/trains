# Every rule pygate knows. Selection, `pygate rule` and the noqa checks all read this table, so
# a code exists exactly when it has a row here.

from __future__ import annotations

from collections.abc import Iterable

from pygate.diagnostic import Rule

RULES: tuple[Rule, ...] = (
    Rule(
        "PG000",
        "syntax-error",
        "The module does not parse.",
        "Every rule reads a module through its syntax tree, so a module that does not parse is "
        "reported rather than skipped: a file no rule could read would otherwise pass them all.",
    ),
    Rule(
        "PG001",
        "unused-noqa",
        "A noqa directive names a pygate rule that suppresses nothing there.",
        "A `# noqa: PGxxx` or `# pygate: noqa: PGxxx` exempts one finding. Once the defect it "
        "excused is fixed, renamed or deleted, the directive stays behind as permanent amnesty for "
        "whatever lands on that line next, so it is reported until it is deleted. Like ruff's "
        "RUF100 it also reports a directive naming a rule that is not enabled for the file. It "
        "cannot itself be silenced by a directive; use `ignore` or `per-file-ignores`.",
    ),
    Rule(
        "PG101",
        "function-body-too-long",
        "A function body runs past `max-function-body-lines`.",
        "A body longer than `[tool.pygate.layout] max-function-body-lines` source lines, its "
        "docstring excluded, is several steps sharing one name. Split it into named helpers: each "
        "gains a name that says what it does and can be exercised on its own. Ruff's PLR0915 "
        "counts statements, which is a proxy; this counts the lines a reader reads.",
    ),
    Rule(
        "PG102",
        "module-docstring",
        "A module carries a module-level docstring.",
        "A module states its contract in code and in comments. Move any real constraint to a "
        "comment or to the definition it governs, and delete the rest. Pairs with ignoring ruff's "
        "D100 and D104.",
    ),
    Rule(
        "PG103",
        "dunder-all-not-last",
        "`__all__` is not the last statement of its module.",
        "`__all__` is the module's export list, and a reader looks for it in one place: the very "
        "bottom of the file.",
    ),
    Rule(
        "PG104",
        "dunder-all-unsorted",
        "`__all__` is not sorted.",
        "Exports stay diffable when `__all__` is ordered: SCREAMING_SNAKE_CASE constants, then "
        "CamelCase classes, then everything else, each group in natural order so that `lane2` "
        "precedes `lane10`.",
    ),
    Rule(
        "PG105",
        "zone-order",
        "A definition is stated out of zone order.",
        "A module reads top-down as its own table of contents: module constants, public classes, "
        "private classes, public functions, private functions, then `__all__`. Imports, "
        "executable setup and lowercase module-level assignments are not governed and stay where "
        "they are. A constant built from a class or function defined above it is exempt, because "
        "hoisting it would be a NameError. `pygate check --fix` reorders them, proving each "
        "rewrite a pure permutation first; `--diff` previews it.",
    ),
    Rule(
        "PG106",
        "nested-function-captures-nothing",
        "An inner function reads nothing from the function it sits in.",
        "An inner function that captures no name from its enclosing function is a module-level "
        "helper hiding from the zone order: untestable on its own, and rebuilt on every call. Lift "
        "it to module level, passing what it reads as arguments. Closures, decorated inner "
        "functions and returned factories are structural and are not reported.",
    ),
    Rule(
        "PG201",
        "global-statement",
        "A `global` statement rebinds a module-level name.",
        "A name rebound at runtime is shared by every importer, holding whatever the last caller "
        "left there. State that outlives a call belongs to an object a caller constructs and "
        "passes: a lazy singleton becomes a constructor argument.",
    ),
    Rule(
        "PG202",
        "in-place-mutation",
        "A function writes through a module-level name.",
        "A subscript store, an augmented assignment, a `del`, or a mutating method such as "
        "`append`, `update` or `setdefault`, reached from inside a function, carries state between "
        "calls that every importer shares. Filling a table between top-level statements is still "
        "building a constant. A registry becomes a value built once and handed down.",
    ),
    Rule(
        "PG203",
        "synchronisation-primitive",
        "A module-level lock, event, condition, semaphore or thread-local.",
        "A module-level guard is evidence of module-level state to guard, so it is reported in its "
        "own right rather than waiting for the state it protects to be recognised.",
    ),
    Rule(
        "PG204",
        "memoising-decorator",
        "`@cache`, `@lru_cache` or `@cached_property` installs a memo table.",
        "The table is keyed by arguments, unbounded, shared by every caller and never "
        "invalidated: global state spelled as a decorator. A memo table becomes a field on the "
        "thing it accelerates.",
    ),
    Rule(
        "PG205",
        "mutable-singleton",
        "A module-level instance of a class that writes its own fields after construction.",
        "Moving `_CACHE = {}` onto a field of a module-level object changes how the state is "
        "spelled, not who shares it, so such an instance counts exactly as the global it replaced. "
        "An instance a caller constructs is not reported: it can be built twice, in two tests, "
        "without them interfering.",
    ),
    Rule(
        "PG301",
        "too-many-instance-attributes",
        "A class assigns more than `max-instance-attributes` slots on `self`.",
        "A class holding that much imperatively assembled state is several collaborators sharing a "
        "constructor. Group the slots that change together into an object it holds, and delegate. "
        "Declared dataclass fields are not counted: a typed, ordered record is the shape this "
        "pushes toward. Threshold: `[tool.pygate.composition] max-instance-attributes`.",
    ),
    Rule(
        "PG302",
        "too-many-methods",
        "A class defines more than `max-methods` methods.",
        "A class that answers that many questions is a module with a constructor, and every method "
        "it gains is one more thing that cannot be exercised without building the whole. Move the "
        "methods that share slots into a collaborator and hold it. Threshold: "
        "`[tool.pygate.composition] max-methods`.",
    ),
    Rule(
        "PG401",
        "multiple-concepts",
        "A module holds two or more disconnected clusters of definitions.",
        "Top-level definitions join when one references another or both reference the same "
        "module-level symbol; imports are not counted. Clusters of at least `[tool.pygate."
        "cohesion] min-cluster-size` that share nothing are that many modules sharing a file, and "
        "the fix is to split it along them. Reported at line 1; exempt a module with "
        "`# pygate: noqa: PG401`.",
    ),
    Rule(
        "PG402",
        "wide-public-surface",
        "A module offers more than `max-public-names` public names.",
        "The surface is `__all__` when the module declares one, and its non-underscore top-level "
        "definitions otherwise. A caller cannot hold a wider one in mind: narrow it with "
        "`__all__`, or move a concept out. Reported at line 1.",
    ),
    Rule(
        "PG403",
        "wide-module",
        "A module declares more than `max-top-level-definitions` definitions.",
        "Width is a smell, not a defect: a wide module may still hold one concept. This rule is "
        'therefore not selected by default; `extend-select = ["PG403"]` lists the widest.',
        default=False,
    ),
)

RULES_BY_CODE: dict[str, Rule] = {rule.code: rule for rule in RULES}

DEFAULT_CODES: frozenset[str] = frozenset(rule.code for rule in RULES if rule.default)


def expand(selectors: Iterable[str]) -> frozenset[str]:
    """Resolve rule codes and code prefixes into the codes they name.

    Args:
        selectors: Codes such as ``PG101``, prefixes such as ``PG1`` or ``PG``, or ``ALL``.

    Returns:
        Every code a selector names.

    Raises:
        ValueError: When a selector names no rule, which is almost always a typo.
    """
    codes: set[str] = set()
    for selector in selectors:
        prefix = "" if selector.upper() == "ALL" else selector.upper()
        matched = {rule.code for rule in RULES if rule.code.startswith(prefix)}
        if not matched:
            raise ValueError(f"unknown rule selector {selector!r}")
        codes |= matched
    return frozenset(codes)
