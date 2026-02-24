## AgenticPMS

`AgenticPMS` is a hierarchical agentic project-management system. It lets you compose:

- **Project agents**: own a project or stream of work, maintain long‑lived context, and make high‑level decisions.
- **Task/worker agents**: execute concrete tasks, call tools, and report status back to the project agent.
- **Workflow definitions**: YAML‑based configurations (see `configurations/`) that wire agents, tools, and memory together without changing code.

### Getting started

**Install dependencies**

```bash
uv run python main.py +name=plan_project 
```

### Typical use cases

- **Agentic project management**: break down goals into plans, tasks, and sub‑tasks managed by specialized agents.
- **Experiment orchestration**: define experiments as agent workflows in YAML and run them without touching Python code.
- **Pluggable tools and memories**: add new tools or memory backends and reference them from configuration files.

See the configuration files under `configurations/` for concrete examples of how `AgenticPMS` composes these pieces.