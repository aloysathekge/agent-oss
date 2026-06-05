"""Skill registry.

Walks every subdirectory of `tools/` that contains a `skill.md`, parses the
frontmatter, and returns a `{skill_name: skill_info}` registry.

A skill is any folder that contains:
  - `skill.md` with YAML frontmatter: `name`, `description`, optional `triggers`
  - a Python package (`__init__.py`) exposing `<NAME>_TOOLS` — a list of
    LangChain `@tool` callables.
  - optionally `<NAME>_TOOLS_FACTORY` — a callable that receives runtime config
    and returns per-request LangChain tools.

Add a new skill = drop in a new folder. No edits here needed.
"""
import importlib
import re
from pathlib import Path

TOOLS_DIR = Path(__file__).parent

_FRONTMATTER_RE = re.compile(r"---\n(.*?)\n---", re.DOTALL)


def _parse_frontmatter(text: str) -> dict:
    """Parse the leading `--- ... ---` YAML block. Minimal parser: `key: value` lines only."""
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}
    meta = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta


def discover_skills() -> dict:
    """Scan `tools/` and return {name: {description, triggers, skill_md_path, tools}}."""
    skills = {}

    for entry in TOOLS_DIR.iterdir():
        if not entry.is_dir() or entry.name.startswith("_"):
            continue

        skill_file = entry / "skill.md"
        if not skill_file.exists():
            continue

        meta = _parse_frontmatter(skill_file.read_text())
        if "name" not in meta:
            continue

        # Import the skill package and pull its exported tool list.
        module = importlib.import_module(f"tools.{entry.name}")
        tools_attr = f"{entry.name.upper()}_TOOLS"
        tools_list = getattr(module, tools_attr, [])
        tools_factory = getattr(module, f"{entry.name.upper()}_TOOLS_FACTORY", None)

        skills[meta["name"]] = {
            "description": meta.get("description", ""),
            "triggers": meta.get("triggers", ""),
            "skill_md_path": str(skill_file),
            "tools": tools_list,
            "tools_factory": tools_factory,
        }

    return skills
