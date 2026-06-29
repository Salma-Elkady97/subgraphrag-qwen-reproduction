"""Resolve ${paths.NAME} tokens used in CORE pipeline scripts.

Path values are read from configs/paths.yaml (or the env var ``SQR_PATHS_YAML``
pointing at an alternative file). Each token in a string of the form
``${paths.NAME}`` is replaced by the resolved value of ``paths.NAME``.

This module exists only so that the CORE scripts (e.g.
``src/reasoning/repair_macro_guarded_v10.py``) can keep their original logic
verbatim while reading their input/output file paths from a single config.
It does NOT change any pipeline behavior.
"""

from __future__ import annotations
import os
import re
import yaml
from pathlib import Path

_TOKEN_RE = re.compile(r"\${paths\.([A-Za-z_][A-Za-z0-9_]*)}")

_DEFAULT_YAML = (
    Path(__file__).resolve().parents[2] / "configs" / "paths.yaml"
)

_cached: dict[str, str] | None = None


def _load() -> dict[str, str]:
    global _cached
    if _cached is not None:
        return _cached
    yaml_path = Path(os.environ.get("SQR_PATHS_YAML", _DEFAULT_YAML))
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    paths = data.get("paths") or {}
    repo_root = str(Path(yaml_path).resolve().parents[1])
    rendered = {}
    for k, v in paths.items():
        rendered[k] = str(v).replace("${repo_root}", repo_root)
    _cached = rendered
    return rendered


def resolve(value: str) -> str:
    """Replace every ``${paths.NAME}`` token in ``value`` with its config value."""
    paths = _load()
    def _sub(m):
        name = m.group(1)
        if name not in paths:
            raise KeyError(f"Path token ${{paths.{name}}} is not defined in configs/paths.yaml")
        return paths[name]
    return _TOKEN_RE.sub(_sub, value)
