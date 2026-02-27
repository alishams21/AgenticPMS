## AgenticPMS

`AgenticPMS` is a fault tolerant hierarchical agentic project-management system. It lets you compose:

- **Project agents**: own a project or stream of work, maintain long‑lived context, and make high‑level decisions.
- **Task/worker agents**: execute concrete tasks, call tools, and report status back to the project agent.
- **Workflow definitions**: YAML‑based configurations (see `configurations/`) that wire agents, tools, and memory together without changing code.

### Features

- **Planner/designer/critic loop**: iterative refinement with dedicated planner, designer, and critic agents.
- **Checkpointing**: in‑memory snapshots of the current plan and critic scores for each run, providing intra‑run fault tolerance when combined with reset tools.
- **Reset-on-degradation**: optional rollback tool that compares the latest scores to the previous checkpoint and, if quality regresses, restores the last good plan while leaving the underlying SQLite session history intact, so bad iterations do not corrupt long‑lived memory.
- **Session persistence**: preserve agent state and context across runs.
- **Run/config wiring**: bind runs to YAML configs so experiments and workflows are driven by configuration.
- **Internal tool-calling flow**: at each stage, agents call internal tools as illustrated in the diagram below.

![Agents internal tool flow](figures/agents_internal.png)

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