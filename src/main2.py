"""
EASY TASK GRADER — Billing Dispute
Difficulty: Easy (target score ~0.7–0.9 for a good agent)

The agent must:
1. Correctly classify the ticket as BILLING / HIGH priority  (+0.25)
2. Draft a professional, empathetic response mentioning the refund  (+0.35)
3. Resolve the ticket with a meaningful summary  (+0.40)

Partial credit awarded at each step.
"""

from __future__ import annotations
from typing import Tuple

from env.models import (
    ActionType, ClassifyAction, DraftResponseAction, ResolveAction,
    EscalateAction, IssueCategory, Priority, EnvironmentState, AgentAction
)
from graders.base import BaseGrader


REFUND_KEYWORDS = {"refund", "credit", "apologize", "apology", "sorry", "billing", "charge", "incorrect"}
EMPATHY_KEYWORDS = {"understand", "inconvenience", "sorry", "apologize", "frustrated", "concern"}


class BillingDisputeGrader(BaseGrader):

    def grade_step(
        self, action: AgentAction, state: EnvironmentState
    ) -> Tuple[float, str, bool]:

        t = action.action_type

        # ---- Classification ----
        if t == ActionType.CLASSIFY:
            action: ClassifyAction
            reward = 0.0
            msgs = []

            if action.category == IssueCategory.BILLING:
                reward += 0.15
                msgs.append("✓ Correct category (billing).")
            else:
                msgs.append(f"✗ Wrong category '{action.category}' — expected 'billing'.")

            if action.priority in (Priority.HIGH, Priority.URGENT):
                reward += 0.10
                msgs.append("✓ Appropriate priority.")
            else:
                msgs.append("✗ Priority should be high/urgent for billing disputes.")

            return reward, " ".join(msgs), False

        # ---- Draft response ----
        if t == ActionType.DRAFT_RESPONSE:
            action: DraftResponseAction
            body_lower = action.body.lower()
            reward = 0.0
            msgs = []

            keyword_hits = sum(1 for k in REFUND_KEYWORDS if k in body_lower)
            empathy_hits = sum(1 for k in EMPATHY_KEYWORDS if k in body_lower)

            if keyword_hits >= 2:
                reward += 0.20
                msgs.append("✓ Response addresses the billing issue.")
            elif keyword_hits == 1:
                reward += 0.10
                msgs.append("~ Response partially addresses billing.")
            else:
                msgs.append("✗ Response doesn't mention billing/refund.")

            if empathy_hits >= 1:
                reward += 0.10
                msgs.append("✓ Empathetic tone detected.")
            else:
                msgs.append("✗ Response lacks empathy.")

            if len(action.body) > 100:
                reward += 0.05
                msgs.append("✓ Response is sufficiently detailed.")

            return reward, " ".join(msgs), False

        # ---- Resolve ----
        if t == ActionType.RESOLVE:
            action: ResolveAction
            reward = 0.0
            msgs = []

            if not state.classification_done:
                msgs.append("✗ Resolved without classifying first (-0.05 penalty).")
                reward -= 0.05
            else:
                reward += 0.10

            if not state.response_drafted:
                msgs.append("✗ Resolved without drafting a response.")
            else:
                reward += 0.15

            if len(action.resolution_summary) > 30:
                reward += 0.10
                msgs.append("✓ Detailed resolution summary.")
            else:
                msgs.append("~ Short resolution summary.")

            msgs.append("Episode complete.")
            return max(0.0, reward), " ".join(msgs), True

        # ---- Unnecessary escalation ----
        if t == ActionType.ESCALATE:
            return 0.0, "✗ Escalation not needed for a standard billing dispute.", True

        # ---- Request info ----
        if t == ActionType.REQUEST_INFO:
            return 0.05, "~ Requested more info (acceptable but not necessary here).", False

        return 0.0, "Unknown action.", False

    def final_score(self, state: EnvironmentState) -> float:
        score = state.cumulative_reward
        # Bonus for completing all 3 stages
        if state.classification_done and state.response_drafted and state.resolved:
            score = min(1.0, score + 0.05)
        return round(min(1.0, max(0.0, score)), 4)