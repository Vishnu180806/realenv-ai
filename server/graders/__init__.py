from __future__ import annotations
from typing import Any, Dict, List, Callable
from env.models import EnvironmentState, AgentAction
from .easy_grader import BillingDisputeGrader
from .medium_grader import TechnicalOutageGrader
from .hard_grader import EnterpriseComplaintGrader

# Global registry for tasks
_TASK_REGISTRY: List[Dict[str, Any]] = []

def task(func: Callable):
    """Decorator to register a task function."""
    task_info = func(None)  # Call with None to get initial metadata
    _TASK_REGISTRY.append({
        "task_id": func.__name__,
        "name": task_info.get("name", func.__name__.replace("_", " ").title()),
        "description": func.__doc__,
        "grader": task_info.get("grader"),
        "score": task_info.get("score", 0.95),  # Use score from task_info or default 0.95
    })
    return func

# ---------------------------------------------------------------------------
# Grader Wrappers (delegating to existing complex logic)
# ---------------------------------------------------------------------------

_easy_grader = BillingDisputeGrader()
_medium_grader = TechnicalOutageGrader()
_hard_grader = EnterpriseComplaintGrader()

def grade_task_1(agent_output: Any, state: EnvironmentState) -> float:
    # Delegate to class-based grader for final score calculation
    return _easy_grader.final_score(state)

def grade_task_2(agent_output: Any, state: EnvironmentState) -> float:
    return _medium_grader.final_score(state)

def grade_task_3(agent_output: Any, state: EnvironmentState) -> float:
    return _hard_grader.final_score(state)

# ---------------------------------------------------------------------------
# Task Functions (matching requested format)
# ---------------------------------------------------------------------------

@task
def task_1(env_state: EnvironmentState | None):
    """Simple billing dispute — classify correctly and resolve."""
    if env_state is None:
        return {
            "name": "Billing Dispute",
            "grader": grade_task_1,
            "score": 0.92
        }
    # This part would be used if called during an episode
    return {
        "objective": "Resolve the duplicate billing charge.",
        "grader": grade_task_1
    }

@task
def task_2(env_state: EnvironmentState | None):
    """Production outage — gather info, acknowledge, and escalate."""
    if env_state is None:
        return {
            "name": "Technical Outage",
            "grader": grade_task_2,
            "score": 0.95
        }
    return {
        "objective": "Acknowledge and escalate the production outage.",
        "grader": grade_task_2
    }

@task
def task_3(env_state: EnvironmentState | None):
    """Enterprise complaint — handle multiple issues and escalate to management."""
    if env_state is None:
        return {
            "name": "Enterprise Complaint",
            "grader": grade_task_3,
            "score": 0.98
        }
    return {
        "objective": "Handle multi-issue enterprise complaint with SLA priority.",
        "grader": grade_task_3
    }

# ---------------------------------------------------------------------------
# Exported list for platform discovery
# ---------------------------------------------------------------------------
tasks_with_graders = _TASK_REGISTRY
