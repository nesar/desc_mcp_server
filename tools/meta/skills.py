"""Server-side skills: named recipes for using the emulator tools well.

A skill is procedural know-how (HOW to combine the tools), as opposed to a
tool (a capability). Each skill is a markdown file in skills/ with a tiny
frontmatter header:

    ---
    name: my-skill
    description: one line shown at planning time
    ---
    ...full instructions...

Progressive disclosure keeps context small: clients list name+description
first and pull the full text only when a task calls for it. The same files
are also registered as native MCP prompts (see mcp_server.server), so
prompt-capable clients get them without any tool call. The file format is
deliberately identical to client-side skill loaders, so the same recipes
can be vendored into an agent framework directly.
"""

from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parents[2] / "skills"


def parse_frontmatter(path: Path) -> dict:
    meta, lines = {}, path.read_text(encoding="utf-8").splitlines()
    if lines and lines[0].strip() == "---":
        for line in lines[1:]:
            if line.strip() == "---":
                break
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    return meta


def skill_files() -> list[Path]:
    return sorted(SKILLS_DIR.glob("*.md"))


def skill_index() -> dict[str, str]:
    """{name: description} for every skill on disk."""
    index = {}
    for path in skill_files():
        meta = parse_frontmatter(path)
        if "name" in meta:
            index[meta["name"]] = meta.get("description", "")
    return index


def skill_text(name: str) -> str | None:
    for path in skill_files():
        if parse_frontmatter(path).get("name") == name:
            return path.read_text(encoding="utf-8")
    return None
