from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
from env.models import (
    EnvironmentState, AgentAction, Ticket, ActionType, ClassifyAction, 
    DraftResponseAction, EscalateAction, IssueCategory, Priority
)
from graders.base import BaseGrader

@dataclass
class StepResult:
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]

class CustomerSupportEnv:
    def __init__(self, task_id: str, ticket: Ticket, grader: BaseGrader, max_steps: int = 10):
        self.task_id = task_id
        self.state = EnvironmentState(ticket=ticket, max_steps=max_steps)
        self.grader = grader

    def reset(self) -> Dict[str, Any]:
        self.state.step = 0
        self.state.actions_taken = []
        self.state.cumulative_reward = 0.0
        self.state.classification_done = False
        self.state.response_drafted = False
        self.state.resolved = False
        self.state.escalated = False
        return self._get_obs()

    def step(self, action: AgentAction) -> StepResult:
        if self.state.resolved or self.state.step >= self.state.max_steps:
            raise RuntimeError("Episode is already finished.")

        self.state.step += 1
        reward, feedback, done = self.grader.grade_step(action, self.state)
        
        # Update state flags
        if action.action_type == ActionType.CLASSIFY:
            self.state.classification_done = True
        elif action.action_type == ActionType.DRAFT_RESPONSE:
            self.state.response_drafted = True
        elif action.action_type == ActionType.ESCALATE:
            self.state.escalated = True
            done = True # Usually escalation ends episode
        elif action.action_type == ActionType.RESOLVE:
            self.state.resolved = True
            done = True

        self.state.cumulative_reward += reward
        self.state.actions_taken.append(action.dict())
        
        if done:
            self.state.resolved = True

        return StepResult(
            observation=self._get_obs(),
            reward=reward,
            done=done,
            info={"feedback": feedback}
        )

    def _get_obs(self) -> Dict[str, Any]:
        return {
            "ticket": self.state.ticket.dict(),
            "step": self.state.step,
            "max_steps": self.state.max_steps,
            "cumulative_reward": self.state.cumulative_reward,
            "actions_taken": self.state.actions_taken,
            "classification_done": self.state.classification_done,
            "response_drafted": self.state.response_drafted,
            "resolved": self.state.resolved,
            "escalated": self.state.escalated
        }

    def final_score(self) -> float:
        return self.grader.final_score(self.state)

    def to_json(self) -> str:
        return self.state.json()

def make_env(task_id: str) -> CustomerSupportEnv:
    from tasks.registry import TASK_REGISTRY
    if task_id not in TASK_REGISTRY:
        raise ValueError(f"Unknown task_id: {task_id}")
    
    config = TASK_REGISTRY[task_id]
    ticket = config["ticket_factory"]()
    grader = config["grader"]
    return CustomerSupportEnv(task_id, ticket, grader, max_steps=config.get("max_steps", 10))
