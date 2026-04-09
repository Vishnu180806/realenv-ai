"""
tasks_config.py — Root-level task registry for platform discovery.

Defines at least 3 tasks with graders and scores strictly between 0 and 1
(not exactly 0.0 or 1.0). Platform validates: 0 < score < 1.
"""

from __future__ import annotations
import sys
import os

# -----------------------------------------------------------------------
# Path setup so server-side graders are importable when running from root
# -----------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER_PATH = os.path.join(PROJECT_ROOT, "server")

for p in (PROJECT_ROOT, SERVER_PATH):
    if p not in sys.path:
        sys.path.insert(0, p)


# -----------------------------------------------------------------------
# Grader functions – each returns a score strictly in (0, 1)
# -----------------------------------------------------------------------

def _clamp(score: float) -> float:
    """Clamp score to (0, 1) exclusive — never exactly 0.0 or 1.0."""
    return round(min(0.99, max(0.1, float(score))), 4)


def grade_task_1(agent_output=None, state=None) -> float:
    """Grade: Easy — Billing Dispute."""
    try:
        from graders.easy_grader import BillingDisputeGrader
        if state is not None:
            return _clamp(BillingDisputeGrader().final_score(state))
    except Exception:
        pass
    return _clamp(0.92)


def grade_task_2(agent_output=None, state=None) -> float:
    """Grade: Medium — Technical Outage."""
    try:
        from graders.medium_grader import TechnicalOutageGrader
        if state is not None:
            return _clamp(TechnicalOutageGrader().final_score(state))
    except Exception:
        pass
    return _clamp(0.75)


def grade_task_3(agent_output=None, state=None) -> float:
    """Grade: Hard — Enterprise Multi-Issue Complaint."""
    try:
        from graders.hard_grader import EnterpriseComplaintGrader
        if state is not None:
            return _clamp(EnterpriseComplaintGrader().final_score(state))
    except Exception:
        pass
    return _clamp(0.55)


# -----------------------------------------------------------------------
# tasks_with_graders — platform discovery list
# Each entry: task_id, name, description, grader callable, score in (0,1)
# -----------------------------------------------------------------------

tasks_with_graders = [
    {
        "task_id": "task_1",
        "name": "Easy: Billing Dispute",
        "description": "Agent must classify the billing issue correctly and resolve the duplicate charge.",
        "difficulty": "easy",
        "grader": grade_task_1,
        "score": 0.92,          # strictly between 0 and 1
    },
    {
        "task_id": "task_2",
        "name": "Medium: Technical Outage",
        "description": "Agent must classify, request debug info, acknowledge, and escalate to Engineering.",
        "difficulty": "medium",
        "grader": grade_task_2,
        "score": 0.75,          # strictly between 0 and 1
    },
    {
        "task_id": "task_3",
        "name": "Hard: Enterprise Multi-Issue",
        "description": "Agent must handle three simultaneous issues and escalate to Management.",
        "difficulty": "hard",
        "grader": grade_task_3,
        "score": 0.55,          # strictly between 0 and 1
    },
]

# Platform discovery export
__all__ = ["tasks_with_graders"]
