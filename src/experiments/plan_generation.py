"""Plan-generation experiment: project (and optionally module/task) planning only."""

import asyncio
import csv
import logging
import uuid
from datetime import datetime
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.experiments.base_experiment import BaseExperiment
from src.project_agents.stateful_project_agent import StatefulProjectAgent
from src.utils.logging import BaseLogger, FileLoggingContext
from src.utils.parallel import run_parallel_isolated
from src.utils.print_utils import bold_green, yellow

console_logger = logging.getLogger(__name__)

# Pipeline stages for plan generation (project → module → task)
PIPELINE_STAGES = ["project", "module", "task"]


def _load_prompts_from_csv(csv_path: str) -> list[tuple[int, str]]:
    """Load prompts from CSV.

    Args:
        csv_path: Path to CSV with columns: index, description (header required).

    Returns:
        List of (index, prompt) tuples.
    """
    prompts_with_ids = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row_num, row in enumerate(reader, start=2):
            if len(row) < 2:
                raise ValueError(f"CSV row {row_num} has fewer than 2 columns: {row}")
            try:
                idx = int(row[0])
            except ValueError:
                raise ValueError(
                    f"CSV row {row_num}: index '{row[0]}' is not a valid integer"
                )
            prompts_with_ids.append((idx, row[1]))
    return prompts_with_ids


def _generate_project_plan_worker(
    prompt: str,
    index: int,
    output_dir: str,
    cfg_dict: dict,
    experiment_run_id: str | None = None,
) -> str | None:
    """Generate a single project plan in an isolated process.

    Top-level function for pickling when using ProcessPoolExecutor.
    All arguments must be picklable.

    Args:
        prompt: Project description.
        index: Prompt index for directory naming.
        output_dir: Base output directory (string path).
        cfg_dict: Resolved config as dict.
        experiment_run_id: Optional run ID for logging.

    Returns:
        Final plan text on success, None on failure (exception is raised).
    """
    from omegaconf import OmegaConf

    out_path = Path(output_dir) / f"prompt_{index:03d}"
    out_path.mkdir(parents=True, exist_ok=True)
    logger = BaseLogger(output_dir=out_path)

    log_path = out_path / "plan.log"
    with FileLoggingContext(log_file_path=log_path, suppress_stdout=True):
        console_logger.info(f"Project plan worker started for prompt {index:03d}")

        cfg = OmegaConf.create(cfg_dict)
        agent = BaseExperiment.build_project_agent(
            cfg_dict=cfg_dict,
            compatible_agents=PlanGenerationExperiment.compatible_project_agents,
            logger=logger,
        )
        plan = asyncio.run(agent.generate_project_plan(prompt=prompt, output_dir=out_path))
        console_logger.info(f"Project plan worker completed for prompt {index:03d}")
        return plan


class PlanGenerationExperiment(BaseExperiment):
    """Experiment that runs project (and optionally module/task) planning only."""

    compatible_project_agents = {
        "workflow_project_agent": StatefulProjectAgent,
    }

    def _run_serial(self, prompts_with_ids: list[tuple[int, str]], cfg_dict: dict) -> None:
        """Run project planning sequentially."""
        console_logger.info("Running project planning serially")

        for index, prompt in prompts_with_ids:
            out_dir = self.output_dir / f"prompt_{index:03d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            logger = BaseLogger(output_dir=out_dir)

            agent = BaseExperiment.build_project_agent(
                cfg_dict=cfg_dict,
                compatible_agents=self.compatible_project_agents,
                logger=logger,
            )
            asyncio.run(agent.generate_project_plan(prompt=prompt, output_dir=out_dir))
            console_logger.info(f"Completed prompt {index:03d}")

    def _run_parallel(
        self,
        prompts_with_ids: list[tuple[int, str]],
        cfg_dict: dict,
        num_workers: int,
        experiment_run_id: str,
    ) -> None:
        """Run project planning in parallel processes."""
        console_logger.info(f"Running project planning in parallel with {num_workers} workers")

        tasks = []
        for index, prompt in prompts_with_ids:
            task_id = f"prompt_{index:03d}"
            tasks.append(
                (
                    task_id,
                    _generate_project_plan_worker,
                    {
                        "prompt": prompt,
                        "index": index,
                        "output_dir": str(self.output_dir),
                        "cfg_dict": cfg_dict,
                        "experiment_run_id": experiment_run_id,
                    },
                )
            )
            console_logger.info(f"Queued {task_id}: {prompt[:60]}...")

        results = run_parallel_isolated(tasks=tasks, max_workers=num_workers)

        failed = [(tid, err) for tid, (ok, err) in results.items() if not ok]
        if failed:
            details = "\n".join(f"  - {tid}: {err}" for tid, err in failed)
            raise RuntimeError(f"{len(failed)}/{len(tasks)} prompt(s) failed:\n{details}")

    def plan_project(self) -> None:
        """Run project planning for all configured prompts (serial or parallel)."""
        pipeline_cfg = self.cfg.experiment.pipeline
        start_stage = pipeline_cfg.start_stage
        stop_stage = pipeline_cfg.stop_stage

        if start_stage not in PIPELINE_STAGES or stop_stage not in PIPELINE_STAGES:
            raise ValueError(
                f"Invalid pipeline stages. start_stage={start_stage!r}, stop_stage={stop_stage!r}. "
                f"Valid: {PIPELINE_STAGES}"
            )
        if PIPELINE_STAGES.index(start_stage) > PIPELINE_STAGES.index(stop_stage):
            raise ValueError(
                f"start_stage '{start_stage}' cannot be after stop_stage '{stop_stage}'"
            )

        # Only run project stage for now
        if start_stage != "project" or stop_stage != "project":
            console_logger.info(
                f"Only project stage is supported; running project. "
                f"(start_stage={start_stage}, stop_stage={stop_stage})"
            )

        csv_path = self.cfg.experiment.get("csv_path")
        if csv_path:
            prompts_with_ids = _load_prompts_from_csv(csv_path)
            console_logger.info(f"Loaded {len(prompts_with_ids)} prompts from CSV: {csv_path}")
        else:
            prompts = self.cfg.experiment.prompts
            prompts_with_ids = list(enumerate(prompts))

        num_workers = min(self.cfg.experiment.num_workers, len(prompts_with_ids))
        experiment_run_id = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        )

        console_logger.info(f"Starting project planning: {num_workers} worker(s), {len(prompts_with_ids)} prompt(s)")
        console_logger.info(f"Experiment run ID: {experiment_run_id}")

        cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)

        if num_workers == 1:
            self._run_serial(prompts_with_ids=prompts_with_ids, cfg_dict=cfg_dict)
        else:
            self._run_parallel(
                prompts_with_ids=prompts_with_ids,
                cfg_dict=cfg_dict,
                num_workers=num_workers,
                experiment_run_id=experiment_run_id,
            )

        console_logger.info("All project plans completed")
        console_logger.info("=" * 60)
        console_logger.info(bold_green("ALL PROJECT PLANS COMPLETED!"))
        console_logger.info("=" * 60)
        console_logger.info(yellow("Outputs saved under: ") + str(self.output_dir))
        console_logger.info("=" * 60)
