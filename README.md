# AgenticPMS

**Agentic Construction of Execution-Ready Project Roadmaps**

`AgenticPMS` is an autonomous, fault-tolerant hierarchical agentic system that transforms high-level project briefs into execution-ready roadmaps through a three-stage pipeline: **project** → **module** → **task**. Each stage is handled by a specialized agent that uses an internal planner/designer/critic loop to refine outputs before passing them to the next level.

---

## What This Project Is

AgenticPMS takes a project description (e.g., "Build a customer analytics dashboard") and produces:

- **Project plan** — High-level components, boundaries, and dependencies
- **Module plan** — Module structure per component (no tasks yet)
- **Task plan** — Concrete tasks per module, ready for execution

The system is **autonomous**: you provide a brief, and agents iteratively refine plans until they meet quality criteria. It is **hierarchical**: each stage consumes the output of the previous one, ensuring alignment from strategy down to tasks.

---

## Pipeline Stages

| Stage         | Input                         | Output                 | Agent                       |
|---------------|-------------------------------|------------------------|-----------------------------|
| Project       | User prompt (brief)           | `project_plan.md`      | StatefulProjectAgent        |
| Module        | `project_plan.md`             | `module_plan.md`      | StatefulModuleAgent         |
| Task          | `module_plan.md`              | `task_plan.md`        | StatefulTaskAgent           |
| Visualization | `project_plan.md` + `module_plan.md` + `task_plan.md` | `hierarchy_diagram.png` (and optionally `.pdf`) | StatefulVisualizationAgent |

Each stage runs independently. The experiment orchestrator chains them in order. The **visualization** stage runs after task and reads all three plan files to produce a hierarchy diagram (Mermaid → PNG/PDF). The visualization critic uses a **vision-capable model (VLM)** to compare the diagram image against the plans for consistency. For PNG/PDF rendering, [mermaid-cli](https://www.npmjs.com/package/@mermaid-js/mermaid-cli) is required. Install it from the project root with:

```bash
npm install
```

Alternatively, the code can use `npx -y @mermaid-js/mermaid-cli` if Node.js and npm are available (no local install needed).

---

## How Agents Work

Every stage agent (project, module, task) uses the same internal architecture: **three sub-agents** coordinated by a planner:

### 1. Designer

Creates and refines the plan. It receives the context (brief or parent plan), produces an initial design, and iteratively improves it based on critic feedback. Its conversation history is persisted in a SQLite session so the critic and subsequent iterations have full context.

### 2. Critic

Evaluates the current plan against domain-specific criteria (structure, clarity, completeness, etc.) and returns structured feedback with scores. The critic reads the designer’s session history to see the current plan. Its output is passed to the planner, which decides whether to request changes or stop.

### 3. Planner

Orchestrates the designer and critic via internal tools:

- `request_initial_design` — Ask the designer for the first plan
- `request_critique` — Ask the critic to evaluate the current plan
- `request_design_change` — Ask the designer to address specific issues (with critic feedback)
- `reset_plan_to_previous_checkpoint` — Roll back if scores degrade (see below)

The planner alternates between critique and design-change until quality thresholds are met or max rounds are reached.

![Agents internal tool flow](figures/agents_internal.png)

---

## Reset-on-Degradation (Checkpoint Rollback)

When the critic loop runs, each iteration updates a checkpoint (current plan + scores). If a later iteration **degrades** the total score, the planner can call `reset_plan_to_previous_checkpoint` to:

- Compare the latest critique scores with the previous checkpoint
- If the total score decreased, restore the last good plan (N-1)
- Update the designer session with the reverted plan so the designer continues from a better state

This keeps bad iterations from corrupting long-lived memory and provides **intra-run fault tolerance**: the system can recover from quality regressions without manual intervention.

---

## Experiment Orchestration

Experiments are driven by Hydra configuration and YAML workflows:

1. **Entry point**: `main.py` with `+name=plan_project` (or another task name)
2. **Config**: `configurations/` defines agents, pipelines, and experiment settings
3. **Pipeline config**: `start_stage` and `stop_stage` control which stages run (e.g., project only, or project → module → task)
4. **Prompts**: Loaded from CSV; each prompt gets its own output directory (`prompt_000`, `prompt_001`, …)
5. **Execution**: Serial or parallel (ProcessPoolExecutor) across prompts

For each prompt, the orchestrator:

1. Runs the project agent → writes `project_plan.md`
2. Runs the module agent with `project_plan.md` as input → writes `module_plan.md`
3. Runs the task agent with `module_plan.md` as input → writes `task_plan.md`

Outputs are written under `outputs/<timestamp>/<run>/prompt_<id>/`.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Autonomous hierarchical system** | Agents operate independently at each level; the pipeline chains them without manual handoffs. |
| **Fault tolerance** | Checkpoint rollback recovers from quality regressions. Session persistence (SQLite) survives restarts. |
| **Consistency between agents** | Each stage consumes the previous stage’s output. The critic loop ensures outputs meet quality standards before advancing. |
| **Critic loop per agent** | Every agent (project, module, task) runs its own designer/critic cycle. Plans are refined to a high standard before being passed to the next level. |
| **Session persistence** | Designer and critic conversations are stored in SQLite (e.g. `prompt_000_project_designer.db`), with optional turn-trimming and summarization for long runs. |
| **Config-driven workflows** | Agents, tools, and pipelines are wired via YAML; no code changes needed for new experiments. |

---

## Getting Started

**Install dependencies**

```bash
uv sync
```

For the **visualization** stage (PNG/PDF diagram output), also install the Mermaid CLI and Chrome for Puppeteer:

```bash
npm install
npx puppeteer browsers install chrome-headless-shell
```

(Node.js ≥18 required; Node ≥20 recommended for mermaid-cli. The second command installs a headless Chrome used by mermaid-cli to render diagrams.)

**Run plan generation** (project → module → task)

```bash
uv run python main.py +name=plan_project
```

**Run project stage only**

```bash
uv run python main.py +name=plan_project experiment.pipeline.start_stage=project experiment.pipeline.stop_stage=project
```

**Run full pipeline including visualization** (project → module → task → hierarchy diagram)

```bash
uv run python main.py +name=plan_project experiment.pipeline.stop_stage=visualization
```

See `configurations/` for pipeline and agent configuration examples.

For details on the visualization stage (how the diagram is produced, where files are written, and why you might not see PNG/PDF), see [docs/VISUALIZATION_AGENT.md](docs/VISUALIZATION_AGENT.md).
