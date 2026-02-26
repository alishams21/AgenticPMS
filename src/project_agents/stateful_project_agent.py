"""Stateful project planning agent using planner/designer/critic workflow.

This agent mirrors the multi-agent pattern used in scene-generation agents, but
operates purely on textual project plans:

- Designer: creates and refines the hierarchical plan (project → modules → tasks)
- Critic: evaluates the plan and suggests improvements with scored feedback
- Planner: orchestrates iterations between designer and critic via tools
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Tuple

from agents import Agent, FunctionTool, Runner, RunResult
from omegaconf import DictConfig

from src.agent_utils.base_stateful_agent import BaseStatefulAgent, AgentType, log_agent_usage
from src.agent_utils.scoring import CritiqueWithScores
from src.project_agents.base_project_agent import BaseProjectAgent
from src.prompts import ProjectAgentPrompts, prompt_registry

console_logger = logging.getLogger(__name__)


class StatefulProjectAgent(BaseStatefulAgent, BaseProjectAgent):
    """Stateful project planning agent using planner/designer/critic workflow."""

    def __init__(self, cfg: DictConfig, logger: Any):
        """Initialize the project planning agent.

        Args:
            cfg: Hydra configuration for the agent.
            logger: Logger / run context with an ``output_dir`` attribute.
        """
        # Initialize both the project-specific base and the shared stateful base.
        BaseProjectAgent.__init__(self, cfg=cfg, logger=logger)
        BaseStatefulAgent.__init__(self, cfg=cfg, logger=logger)

        # Persistent agent sessions (reuse BaseStatefulAgent implementation).
        self.designer_session, self.critic_session = self._create_sessions()

        # Agent instances (created lazily when running a plan).
        self.designer: Agent | None = None
        self.critic: Agent | None = None
        self.planner: Agent | None = None

        # Project prompt describing the overall plan (set when generating a plan).
        self.project_prompt: str = ""

    # -------------------------------------------------------------------------
    # BaseStatefulAgent abstract interface implementations
    # -------------------------------------------------------------------------

    @property
    def agent_type(self) -> AgentType:
        """Project planning agent type for the BaseStatefulAgent interface."""
        return AgentType.PROJECT

    def _get_final_scores_directory(self) -> Path:
        """Directory where any final evaluation artifacts would be stored."""
        return Path(self.logger.output_dir) / "final_project_plan"

    def _get_critique_prompt_enum(self) -> Any:
        """Prompt enum for critic runner instruction (unused by current tools)."""
        return ProjectAgentPrompts.CRITIC_RUNNER_INSTRUCTION

    def _get_design_change_prompt_enum(self) -> Any:
        """Prompt enum for design-change instruction (unused by current tools)."""
        return ProjectAgentPrompts.DESIGNER_CRITIQUE_INSTRUCTION

    def _get_initial_design_prompt_enum(self) -> Any:
        """Prompt enum for initial design instruction (unused by current tools)."""
        return ProjectAgentPrompts.DESIGNER_INITIAL_INSTRUCTION

    def _get_initial_design_prompt_kwargs(self) -> dict:
        """Template variables for initial design prompt (none for project agent)."""
        return {}

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _create_designer_agent(self, tools: list[FunctionTool] | None = None) -> Agent:
        """Create the project designer agent.

        Mirrors the pattern used in other stateful agents by delegating to
        BaseStatefulAgent._create_designer_agent, while keeping the existing
        project-specific prompt configuration.
        """
        if tools is None:
            tools = []

        return super()._create_designer_agent(
            tools=tools,
            prompt_enum=ProjectAgentPrompts.DESIGNER_AGENT,
            project_prompt=self.project_prompt,
        )

    def _create_critic_agent(self, tools: list[FunctionTool] | None = None) -> Agent:
        """Create the project critic agent.

        Mirrors the pattern used in other stateful agents by delegating to
        BaseStatefulAgent._create_critic_agent, while keeping the existing
        project-specific prompt configuration.
        """
        if tools is None:
            tools = []

        return super()._create_critic_agent(
            tools=tools,
            prompt_enum=ProjectAgentPrompts.CRITIC_AGENT,
            output_type=CritiqueWithScores,
            project_prompt=self.project_prompt,
        )

    def _create_planner_tools(self) -> list[FunctionTool]:
        """Create planner tools using the shared BaseStatefulAgent implementation."""
        return super()._create_planner_tools()

    def _create_planner_agent(self, tools: list[FunctionTool]) -> Agent:
        """Create the planner agent that orchestrates designer and critic.

        Delegates to BaseStatefulAgent._create_planner_agent so the planner
        shares the same configuration pattern (model settings, etc.) as other
        stateful agents while using project-specific prompts.
        """
        max_rounds = getattr(self.cfg, "max_critique_rounds", 3)
        early_finish_min_score = getattr(self.cfg, "early_finish_min_score", 8)

        return super()._create_planner_agent(
            tools=tools,
            prompt_enum=ProjectAgentPrompts.PLANNER_AGENT,
            project_prompt=self.project_prompt,
            max_critique_rounds=max_rounds,
            early_finish_min_score=early_finish_min_score,
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    async def generate_project_plan(self, prompt: str, output_dir: Path) -> str:
        """Generate a hierarchical project plan for the given prompt.

        This method:
        1. Initializes designer, critic, and planner agents
        2. Runs the planner with its runner instruction
        3. Saves the final plan to ``output_dir`` and returns it
        """
        console_logger.info("Starting project planning workflow")

        self.project_prompt = prompt
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create agents.
        self.designer = self._create_designer_agent()
        self.critic = self._create_critic_agent()
        planner_tools = self._create_planner_tools()
        self.planner = self._create_planner_agent(tools=planner_tools)

        # Get runner instruction.
        runner_instruction = prompt_registry.get_prompt(
            prompt_enum=ProjectAgentPrompts.PLANNER_RUNNER_INSTRUCTION,
        )

        # Run planning workflow.
        planner_max_turns = getattr(
            getattr(self.cfg.agents, "planner_agent", None), "max_turns", 64
        )
        result: RunResult = await Runner.run(
            starting_agent=self.planner,
            input=runner_instruction,
            max_turns=planner_max_turns,
        )

        log_agent_usage(result=result, agent_name="PLANNER (PROJECT)")

        final_plan = result.final_output or ""

        # Save final plan to disk.
        plan_path = output_dir / "project_plan.md"
        plan_path.write_text(final_plan, encoding="utf-8")
        console_logger.info(f"Project plan saved to: {plan_path}")

        return final_plan

