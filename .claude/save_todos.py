#!/usr/bin/env python3
"""PostToolUse hook — writes current TodoWrite state to TODO.md."""

import json
import os
import sys

data = json.load(sys.stdin)
todos = data.get("tool_input", {}).get("todos", [])

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
todo_path = os.path.join(project_root, "TODO.md")

if not todos:
    if os.path.exists(todo_path):
        os.remove(todo_path)
    sys.exit(0)

lines = ["# Pending Work\n\n"]
pending = [t for t in todos if t.get("status") != "completed"]
in_progress = [t for t in pending if t.get("status") == "in_progress"]
waiting = [t for t in pending if t.get("status") == "pending"]

if in_progress:
    lines.append("### In Progress\n")
    for t in in_progress:
        lines.append(f"- [ ] {t['content']}\n")
    lines.append("\n")
if waiting:
    lines.append("### Pending\n")
    for t in waiting:
        lines.append(f"- [ ] {t['content']}\n")
    lines.append("\n")

with open(todo_path, "w", encoding="utf-8") as f:
    f.writelines(lines)
