"""Planner tools for project planning agents.

These tools are called by the planner agent to coordinate the project designer
and critic agents. They mirror the pattern used in floor plan tools but operate
purely on text-based project plans.
"""

from __future__ import annotations

import logging
from typing import Any

from agents import Agent, FunctionTool, Runner, RunResult, function_tool
from agents.memory.session import SQLiteSession
from omegaconf import DictConfig

from src.prompts import ProjectAgentPrompts, prompt_registry

console_logger = logging.getLogger(__name__)


class ProjectPlanTools:
    """Tools for orchestrating project designer and critic agents.

    Exposes a ``tools`` dict mapping tool names to `FunctionTool` instances:
    - request_initial_project_design
    - request_project_critique
    - request_project_design_change
    """

    def __init__(
        self,
        *,
        cfg: DictConfig,
        project_prompt: str,
        designer: Agent,
        critic: Agent,
        designer_session: SQLiteSession,
        critic_session: SQLiteSession,
    ) -> None:
        self.cfg = cfg
        self.project_prompt = project_prompt
        self.designer = designer
        self.critic = critic
        self.designer_session = designer_session
        self.critic_session = critic_session

        self.tools: dict[str, FunctionTool] = self._create_tool_closures()

    def _create_tool_closures(self) -> dict[str, FunctionTool]:
        """Create planner tools as closures over this instance."""

        @function_tool
        async def request_initial_project_design() -> str:
            """Request the designer to create the initial project plan."""
            console_logger.info("Tool called: request_initial_project_design")

            instruction = prompt_registry.get_prompt(
                prompt_enum=ProjectAgentPrompts.DESIGNER_INITIAL_INSTRUCTION,
            )

            result: RunResult = await Runner.run(
                starting_agent=self.designer,
                input=instruction,
                session=self.designer_session,
                max_turns=self.cfg.agents.designer_agent.max_turns,
            )

            return result.final_output or ""

        @function_tool
        async def request_project_critique() -> str:
            """Request the critic to evaluate the current project plan."""
            console_logger.info("Tool called: request_project_critique")

            instruction = prompt_registry.get_prompt(
                prompt_enum=ProjectAgentPrompts.CRITIC_RUNNER_INSTRUCTION,
            )

            result: RunResult = await Runner.run(
                starting_agent=self.critic,
                input=instruction,
                session=self.critic_session,
                max_turns=self.cfg.agents.critic_agent.max_turns,
            )

            return result.final_output or ""

        @function_tool
        async def request_project_design_change(instruction: str) -> str:
            """Request the designer to refine the plan based on critic feedback."""
            console_logger.info("Tool called: request_project_design_change")

            full_instruction = prompt_registry.get_prompt(
                prompt_enum=ProjectAgentPrompts.DESIGNER_CRITIQUE_INSTRUCTION,
                instruction=instruction,
            )

            result: RunResult = await Runner.run(
                starting_agent=self.designer,
                input=full_instruction,
                session=self.designer_session,
                max_turns=self.cfg.agents.designer_agent.max_turns,
            )

            return result.final_output or ""

        return {
            "request_initial_project_design": request_initial_project_design,
            "request_project_critique": request_project_critique,
            "request_project_design_change": request_project_design_change,
        }

