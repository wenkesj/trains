# Configuration: which pyproject.toml governs a file, and what it selects. The shape is ruff's on
# purpose -- `select`, `extend-select`, `ignore`, `per-file-ignores`, `exclude` and
# `extend-exclude` under `[tool.pygate]`, and one table of thresholds per rule family -- so that a
# reader who knows one knows the other. As in ruff, a file is governed by the nearest
# pyproject.toml above it that has the tool table, and patterns are relative to that file.

from __future__ import annotations

import dataclasses
import fnmatch
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pygate.registry import DEFAULT_CODES, expand
from pygate.rules.cohesion import CohesionSettings
from pygate.rules.composition import CompositionSettings
from pygate.rules.layout import LayoutSettings

CONFIG_FILE = "pyproject.toml"

#: Directories no project means to lint. Replaced by `exclude`, extended by `extend-exclude`.
DEFAULT_EXCLUDE: tuple[str, ...] = (
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
)

TOP_LEVEL_KEYS = frozenset(
    {
        "cohesion",
        "composition",
        "exclude",
        "extend-exclude",
        "extend-select",
        "ignore",
        "layout",
        "per-file-ignores",
        "select",
    }
)


class ConfigError(ValueError):
    """A configuration pygate cannot honour. Reported, never guessed around."""


@dataclass(frozen=True)
class Overrides:
    """Selection given on the command line, applied over every file's configuration.

    Attributes:
        select: Replaces the configured ``select`` when given.
        ignore: Added to the configured ``ignore``.
    """

    select: tuple[str, ...] | None = None
    ignore: tuple[str, ...] = ()


@dataclass(frozen=True)
class Settings:
    """Everything one configuration decides.

    Attributes:
        root: Directory patterns are relative to: the one holding the pyproject.toml.
        selected: Codes enabled before per-file ignores.
        per_file_ignores: Glob patterns paired with the codes they disable.
        exclude: Glob patterns for files and directories never checked.
        layout: Thresholds for PG1xx.
        composition: Thresholds for PG3xx.
        cohesion: Thresholds for PG4xx.
    """

    root: Path
    selected: frozenset[str]
    per_file_ignores: tuple[tuple[str, frozenset[str]], ...]
    exclude: tuple[str, ...]
    layout: LayoutSettings
    composition: CompositionSettings
    cohesion: CohesionSettings

    def is_excluded(self, path: Path) -> bool:
        """Decide whether a file, or any directory above it, matches an exclude pattern.

        Args:
            path: File to test.

        Returns:
            Whether the file is excluded.
        """
        parts = _relative(path, self.root).split("/")
        prefixes = ["/".join(parts[: index + 1]) for index in range(len(parts))]
        return any(
            _glob_matches(prefix, pattern) for prefix in prefixes for pattern in self.exclude
        )

    def enabled_for(self, path: Path) -> frozenset[str]:
        """Resolve the codes enabled for one file.

        Args:
            path: File to check.

        Returns:
            The selected codes, less those its per-file ignores disable.
        """
        relative = _relative(path, self.root)
        ignored = {
            code
            for pattern, codes in self.per_file_ignores
            if _glob_matches(relative, pattern)
            for code in codes
        }
        return self.selected - ignored


class Configurations:
    """Settings for each file, from the nearest pyproject.toml above it with ``[tool.pygate]``.

    Note:
        The lookup is cached per directory on the instance, so a run over thousands of files reads
        each configuration once and two runs in one process never share a stale answer.
    """

    def __init__(self, overrides: Overrides, fallback_root: Path) -> None:
        """Prepare an empty cache.

        Args:
            overrides: Command-line selection applied over every configuration.
            fallback_root: Root for defaults, when no configuration governs a file.
        """
        self._overrides = overrides
        self._fallback_root = fallback_root
        self._by_directory: dict[Path, Settings] = {}

    def for_path(self, path: Path) -> Settings:
        """Find the settings that govern a file or directory.

        Args:
            path: File or directory.

        Returns:
            Its settings.

        Raises:
            ConfigError: When the governing configuration is invalid.
        """
        directory = (path if path.is_dir() else path.parent).resolve()
        if directory not in self._by_directory:
            self._by_directory[directory] = self._nearest(directory)
        return self._by_directory[directory]

    def _nearest(self, directory: Path) -> Settings:
        """Walk up from a directory to the first configuration that has the tool table.

        Args:
            directory: Where to start.

        Returns:
            The settings found, or defaults rooted at the fallback root.
        """
        for candidate in (directory, *directory.parents):
            if candidate in self._by_directory:
                return self._by_directory[candidate]
            config = candidate / CONFIG_FILE
            table = _tool_table(config) if config.is_file() else None
            if table is not None:
                return _load_settings(table, candidate, self._overrides, config)
        return _load_settings({}, self._fallback_root.resolve(), self._overrides, None)


def _load_settings(
    table: dict[str, Any], root: Path, overrides: Overrides, source: Path | None
) -> Settings:
    """Build settings from one ``[tool.pygate]`` table.

    Args:
        table: The table's contents.
        root: Directory its patterns are relative to.
        overrides: Command-line selection applied over it.
        source: The file it came from, for error messages.

    Returns:
        The settings.

    Raises:
        ConfigError: When a key is unknown, a value has the wrong type, or a selector names no rule.
    """
    try:
        return _build(table, root, overrides)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{source or 'default settings'}: {exc}") from exc


def _build(table: dict[str, Any], root: Path, overrides: Overrides) -> Settings:
    """Validate a table and resolve its selection.

    Args:
        table: The ``[tool.pygate]`` contents.
        root: Directory its patterns are relative to.
        overrides: Command-line selection.

    Returns:
        The settings.
    """
    unknown = sorted(set(table) - TOP_LEVEL_KEYS)
    if unknown:
        raise ValueError(f"unknown key(s) in [tool.pygate]: {', '.join(unknown)}")
    select = overrides.select if overrides.select is not None else table.get("select")
    base = DEFAULT_CODES if select is None else expand(_strings(select, "select"))
    extended = expand(_strings(table.get("extend-select", []), "extend-select"))
    ignored = expand([*_strings(table.get("ignore", []), "ignore"), *overrides.ignore])
    return Settings(
        root=root,
        selected=(base | extended) - ignored,
        per_file_ignores=_per_file_ignores(table.get("per-file-ignores", {})),
        exclude=(
            *_strings(table.get("exclude", DEFAULT_EXCLUDE), "exclude"),
            *_strings(table.get("extend-exclude", []), "extend-exclude"),
        ),
        layout=_group(LayoutSettings, table.get("layout", {}), "layout"),
        composition=_group(CompositionSettings, table.get("composition", {}), "composition"),
        cohesion=_group(CohesionSettings, table.get("cohesion", {}), "cohesion"),
    )


def _glob_matches(relative: str, pattern: str) -> bool:
    """Match a root-relative path against one pattern, the way ruff reads it.

    Args:
        relative: Path relative to the configuration root, with forward slashes.
        pattern: A glob. ``*`` crosses directories. A pattern with no slash also matches the last
            component alone, so ``build`` excludes a directory of that name at any depth.

    Returns:
        Whether the path matches.
    """
    pattern = pattern.removeprefix("./").rstrip("/")
    if fnmatch.fnmatchcase(relative, pattern):
        return True
    if pattern.startswith("**/") and fnmatch.fnmatchcase(relative, pattern[3:]):
        return True
    return "/" not in pattern and fnmatch.fnmatchcase(relative.rsplit("/", 1)[-1], pattern)


def _group[T](cls: type[T], table: Any, name: str) -> T:
    """Read one family's threshold table into its settings record.

    Args:
        cls: The family's settings dataclass; every field is a positive integer.
        table: The ``[tool.pygate.<name>]`` contents.
        name: The family, for error messages.

    Returns:
        The settings, defaults filling what the table leaves out.
    """
    if not isinstance(table, dict):
        raise TypeError(f"[tool.pygate.{name}] must be a table")
    if not dataclasses.is_dataclass(cls):
        raise TypeError("Settings groups must be dataclasses")
    fields = {field.name.replace("_", "-"): field.name for field in dataclasses.fields(cls)}
    unknown = sorted(set(table) - set(fields))
    if unknown:
        raise ValueError(f"unknown key(s) in [tool.pygate.{name}]: {', '.join(unknown)}")
    for key, value in table.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise TypeError(f"[tool.pygate.{name}] {key} must be a positive integer")
    return cls(**{fields[key]: value for key, value in table.items()})


def _per_file_ignores(value: Any) -> tuple[tuple[str, frozenset[str]], ...]:
    """Read ``per-file-ignores`` into pattern and code pairs.

    Args:
        value: The table's contents.

    Returns:
        Each pattern paired with the codes it disables.
    """
    if not isinstance(value, dict):
        raise TypeError("per-file-ignores must be a table of glob = [codes]")
    return tuple(
        (pattern, expand(_strings(codes, f"per-file-ignores.{pattern!r}")))
        for pattern, codes in value.items()
    )


def _relative(path: Path, root: Path) -> str:
    """Render a path relative to a configuration root.

    Args:
        path: Path to render.
        root: The root.

    Returns:
        The relative path with forward slashes, or the absolute one when it is not beneath root.
    """
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def _strings(value: Any, key: str) -> tuple[str, ...]:
    """Require a list of strings.

    Args:
        value: The configured value.
        key: Its key, for the error message.

    Returns:
        The strings.
    """
    if isinstance(value, str) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{key} must be a list of strings")
    return tuple(value)


def _tool_table(config: Path) -> dict[str, Any] | None:
    """Read a pyproject.toml's ``[tool.pygate]`` table.

    Args:
        config: The file.

    Returns:
        The table, or ``None`` when the file has none.

    Raises:
        ConfigError: When the file is not valid TOML.
    """
    try:
        document = tomllib.loads(config.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{config}: {exc}") from exc
    table = document.get("tool", {}).get("pygate")
    return table if isinstance(table, dict) else None
