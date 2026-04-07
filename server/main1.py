"""
CustomerSupportEnv — OpenEnv-compliant environment.

Simulates a real-world customer support agent workflow:
  - Classify incoming support tickets
  - Draft appropriate responses
  - Escalate when necessary
  - Resolve tickets with quality grading

Implements the full OpenEnv interface:
  reset() -> observation dict
  step(action) -> StepResult
  state() -> EnvironmentState
"""

from __future__ import annotations
import json
import uuid
from typing import Any, Dict, Optional, Type

from env.models import (
    ActionType, AgentAction, ClassifyAction, DraftResponseAction,
    EscalateAction, RequestInfoAction, ResolveAction,
    EnvironmentState, StepResult, Ticket
)
from tasks.registry import TASK_REGISTRY


class CustomerSupportEnv:
    """
    OpenEnv-compliant Customer Support Resolution Environment.

    Action Space:
        Union of ClassifyAction | DraftResponseAction | EscalateAction |
                RequestInfoAction | ResolveAction

    Observation Space:
        Dict containing ticket details, conversation history,
        step number, and available actions.

    Reward:
        Partial credit at each step. Final score 0.0–1.0.
    """

    metadata = {
        "name": "customer-support-v1",
        "version": "1.0.0",
        "render_modes": ["json"],
    }

    def __init__(self, task_id: str, max_steps: int = 10):
        if task_id not in TASK_REGISTRY:
            raise ValueError(f"Unknown task_id '{task_id}'. Available: {list(TASK_REGISTRY.keys())}")
        self.task_id = task_id
        self.max_steps = max_steps
        self._task_def = TASK_REGISTRY[task_id]
        self._state: Optional[EnvironmentState] = None

    # ------------------------------------------------------------------
    # OpenEnv core interface
    # ------------------------------------------------------------------

    def reset(self) -> Dict[str, Any]:
        """Reset environment and return initial observation."""
        ticket: Ticket = self._task_def["ticket_factory"]()
        self._state = EnvironmentState(
            task_id=self.task_id,
            ticket=ticket,
            step_number=0,
            max_steps=self.max_steps,
        )
        return self._build_observation("Environment reset. New ticket assigned.")

    def step(self, action: AgentAction) -> StepResult:
        """
        Execute one agent action and return (observation, reward, done, info).

        Reward is incremental — partial progress is always rewarded.
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        self._state.step_number += 1
        action_dict = action.model_dump()
        self._state.actions_taken.append(action_dict)

        # Route to grader
        grader = self._task_def["grader"]
        step_reward, feedback, done = grader.grade_step(action, self._state)

        # Apply state transitions
        self._apply_state_transition(action)

        # Check terminal conditions
        if self._state.step_number >= self._state.max_steps:
            done = True
            feedback += " [Max steps reached]"

        self._state.cumulative_reward = min(
            1.0, self._state.cumulative_reward + step_reward
        )
        self._state.done = done

        obs = self._build_observation(feedback)
        return StepResult(
            observation=obs,
            reward=step_reward,
            done=done,
            info={
                "cumulative_reward": self._state.cumulative_reward,
                "step": self._state.step_number,
                "feedback": feedback,
                "action_type": action.action_type,
            },
        )

    def state(self) -> EnvironmentState:
        """Return the full current environment state."""
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return self._state

    def final_score(self) -> float:
        """Return the final normalised score (0.0–1.0)."""
        if self._state is None:
            return 0.0
        grader = self._task_def["grader"]
        return grader.final_score(self._state)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_state_transition(self, action: AgentAction):
        s = self._state
        t = action.action_type
        if t == ActionType.CLASSIFY:
            s.classification_done = True
        elif t == ActionType.DRAFT_RESPONSE:
            s.response_drafted = True
        elif t == ActionType.ESCALATE:
            s.escalated = True
            s.done = True
        elif t == ActionType.RESOLVE:
            s.resolved = True
            s.done = True

    def _build_observation(self, message: str) -> Dict[str, Any]:
        s = self._state
        return {
            "message": message,
            "ticket": s.ticket.model_dump(),
            "step": s.step_number,
            "max_steps": s.max_steps,
            "classification_done": s.classification_done,
            "response_drafted": s.response_drafted,
            "resolved": s.resolved,
            "escalated": s.escalated,
            "actions_taken": s.actions_taken,
            "available_actions": [a.value for a in ActionType],
            "cumulative_reward": s.cumulative_reward,
        }

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        return self._state.model_dump_json(indent=2) if self._state else "{}"


# ------------------------------------------------------------------
# Factory function (convenience)
# ------------------------------------------------------------------

def make_env(task_id: str, **kwargs) -> CustomerSupportEnv:
    return CustomerSupportEnv(task_id=task_id, **kwargs)