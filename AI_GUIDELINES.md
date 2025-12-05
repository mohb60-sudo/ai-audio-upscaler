# AI Agent Guidelines

> [!IMPORTANT]
> **ATTENTION AI AGENT**: Before making ANY changes to this codebase, you MUST read and follow the **Safe Edit Protocol**.

## 🚨 Mandatory Workflow
This project enforces a strict safety protocol to prevent regressions and accidental deletions.

1.  **Read the Protocol**: `.agent/workflows/safe_edit.md`
2.  **Use the Tools**: We have built a suite of safety tools in `tools/`.
    *   `tools/verify_change.py`: Run this after EVERY edit.
    *   `tools/map_dependencies.py`: Run this BEFORE editing to understand impact.

## ⛔ Critical Rules
1.  **Never** edit `app.py` or `pipeline.py` without a backup.
2.  **Never** mark a task as complete if `tools/verify_change.py` fails.
3.  **Never** delete UI elements without explicit user approval (check `tools/compare_ui_tree.py`).

## 🛠️ Quick Start
To verify your changes, simply run:
```bash
python tools/verify_change.py <target_file>
```

If this command fails, **REVERT IMMEDIATELY**.
