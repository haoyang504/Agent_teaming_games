"""Shared utilities for the Moon Survival agent system."""

from typing import Dict, List


def format_messages_for_log(messages: List[Dict[str, str]]) -> str:
    """Render an OpenAI-format message list as a top-to-bottom transcript.

    Each block is tagged with its role; user messages with a `name=` field get
    `[user name="X"]` instead of `[user]`. Block content is dumped verbatim
    under each tag.

    Mirrors the format Weikai uses for his prompt-dump files.
    """
    lines: List[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        name = msg.get("name")
        tag = f"[{role} name=\"{name}\"]" if name else f"[{role}]"
        lines.append(tag)
        lines.append(msg.get("content", ""))
        lines.append("")  # blank line between blocks
    return "\n".join(lines)
