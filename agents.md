# Repository Operational Rules & Architecture

## 1. Context Engineering Boundaries
- We use a JIT (Just-In-Time) retrieval model. Never guess architecture.
- When searching documentation, always run: `python query.py --name <project> --agent "your query"`
- The `--agent` flag must return lean, machine-readable XML/Markdown. Strip out decorative terminal UI elements to save tokens.

## 2. Think Before Coding
- State your technical assumptions clearly before writing or modifying any scripts.
- If a RAG search returns ambiguous or conflicting rules, stop and escalate to the user instead of guessing.

## 3. Simplicity & Surgical Changes
- Write the absolute minimum code required to solve a task. Avoid speculative abstractions.
- Touch ONLY the precise lines of code required. Do not reformat or alter adjacent code blocks.

## 4. Persistent Memory Logs
- Maintain a local file named `agent_log.md`.
- After completing a major code change or database rebuild, append a single, timestamped sentence summarizing the action and result so you can maintain state across session resets.
