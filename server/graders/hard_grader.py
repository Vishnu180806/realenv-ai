"""
HARD TASK GRADER — Enterprise Multi-Issue Complaint
Difficulty: Hard (target score ~0.35–0.60 for a good agent)

Scenario: Enterprise customer with THREE simultaneous issues:
  - Unexpected invoice charges (billing)
  - API integration broken (technical)
  - Account access revoked for 2 users (account)

The agent must:
1. Classify as BILLING (primary) with URGENT priority         (+0.15)
2. Request specific info for each sub-issue                   (+0.15)
3. Draft a response that addresses ALL THREE issues           (+0.25)
4. Escalate to MANAGEMENT (enterprise accounts need it)       (+0.20)
5. Provide detailed internal notes with a resolution plan     (+0.25)

This is deliberately hard — a weak agent will miss sub-issues
or escalate to the wrong team.
"""

from __future__ import annotations
from typing import Tuple

from env.models import (
    ActionType, ClassifyAction, DraftResponseAction, ResolveAction,
    EscalateAction, RequestInfoAction, IssueCategory, Priority,
    EscalationTeam, EnvironmentState, AgentAction
)
from graders.base import BaseGrader


BILLING_WORDS = {"invoice", "charge", "payment", "billing", "fee", "cost", "overcharged"}
TECHNICAL_WORDS = {"api", "integration", "broken", "error", "endpoint", "authentication", "token"}
ACCOUNT_WORDS = {"access", "user", "permission", "login", "account", "revoked", "locked"}
PLAN_WORDS = {"billing team", "engineering", "account manager", "24 hours", "priority", "investigate", "review"}


def _coverage(text: str, keywords: set) -> int:
    return sum(1 for k in keywords if k in text.lower())


class EnterpriseComplaintGrader(BaseGrader):

    def grade_step(
        self, action: AgentAction, state: EnvironmentState
    ) -> Tuple[float, str, bool]:

        t = action.action_type

        if t == ActionType.CLASSIFY:
            reward = 0.0
            msgs = []

            if action.category == IssueCategory.BILLING:
                reward += 0.08
                msgs.append("✓ Primary category correct (billing).")
            else:
                msgs.append(f"~ Category '{action.category}' — billing is primary (multi-issue ticket).")
                reward += 0.03  # partial

            if action.priority == Priority.URGENT:
                reward += 0.07
                msgs.append("✓ Urgent priority correct for enterprise.")
            elif action.priority == Priority.HIGH:
                reward += 0.04
                msgs.append("~ High priority okay, but enterprise warrants urgent.")
            else:
                msgs.append("✗ Priority too low for enterprise complaint.")

            return max(0.01, reward), " ".join(msgs), False

        if t == ActionType.REQUEST_INFO:
            combined = " ".join(action.questions).lower()
            billing_c = _coverage(combined, BILLING_WORDS)
            tech_c = _coverage(combined, TECHNICAL_WORDS)
            acct_c = _coverage(combined, ACCOUNT_WORDS)

            covered = sum(1 for c in [billing_c, tech_c, acct_c] if c > 0)
            reward = covered * 0.05
            msgs = [f"Info request covers {covered}/3 issue categories."]
            if covered == 3:
                msgs.append("✓ All sub-issues addressed in clarification.")
            elif covered == 2:
                msgs.append("~ Missing one sub-issue in clarification questions.")
            else:
                msgs.append("✗ Most sub-issues not addressed in info request.")

            return max(0.01, reward), " ".join(msgs), False

        if t == ActionType.DRAFT_RESPONSE:
            body = action.body.lower()
            billing_c = _coverage(body, BILLING_WORDS)
            tech_c = _coverage(body, TECHNICAL_WORDS)
            acct_c = _coverage(body, ACCOUNT_WORDS)
            covered = sum(1 for c in [billing_c, tech_c, acct_c] if c > 0)

            reward = 0.0
            msgs = [f"Response mentions {covered}/3 sub-issues."]

            if covered == 3:
                reward += 0.20
                msgs.append("✓ Excellent — all three issues acknowledged.")
            elif covered == 2:
                reward += 0.12
                msgs.append("~ Two of three issues mentioned.")
            elif covered == 1:
                reward += 0.06
                msgs.append("✗ Only one issue mentioned — enterprise client has 3.")
            else:
                msgs.append("✗ Response doesn't address any specific issue.")

            # Length check for enterprise
            if len(action.body) > 200:
                reward += 0.05
                msgs.append("✓ Appropriately detailed response for enterprise.")

            return max(0.01, reward), " ".join(msgs), False

        if t == ActionType.ESCALATE:
            reward = 0.0
            msgs = []

            if action.team == EscalationTeam.MANAGEMENT:
                reward += 0.15
                msgs.append("✓ Correct — management escalation for enterprise accounts.")
            elif action.team == EscalationTeam.BILLING_TEAM:
                reward += 0.08
                msgs.append("~ Billing team escalation is partial — management needed for enterprise.")
            elif action.team == EscalationTeam.ENGINEERING:
                reward += 0.05
                msgs.append("~ Engineering escalation handles only 1 of 3 sub-issues.")
            else:
                msgs.append("✗ Wrong escalation team for enterprise complaint.")

            notes_lower = action.internal_notes.lower()
            plan_hits = _coverage(notes_lower, PLAN_WORDS)
            if plan_hits >= 3:
                reward += 0.15
                msgs.append("✓ Detailed internal notes with resolution plan.")
            elif plan_hits >= 1:
                reward += 0.08
                msgs.append("~ Internal notes present but lack detail.")
            else:
                msgs.append("✗ No meaningful internal notes provided.")

            msgs.append("Escalated. Episode complete.")
            return max(0.01, reward), " ".join(msgs), True

        if t == ActionType.RESOLVE:
            # Hard task: enterprise multi-issue MUST be escalated
            return 0.01, "✗ Enterprise multi-issue complaint cannot be self-resolved. Escalation required.", True

        return 0.01, "Unknown action.", False

    def final_score(self, state: EnvironmentState) -> float:
        score = state.cumulative_reward
        # Bonus: full pipeline (classify → request_info → draft → escalate)
        action_types_used = {a.get("action_type") for a in state.actions_taken}
        full_pipeline = {
            ActionType.CLASSIFY, ActionType.REQUEST_INFO,
            ActionType.DRAFT_RESPONSE, ActionType.ESCALATE
        }
        if {a.value for a in full_pipeline}.issubset(action_types_used):
            score = min(1.0, score + 0.10)
            
        # Platform requires score strictly between 0 and 1.
        score = max(0.001, min(0.999, score))
        return round(score, 4)
