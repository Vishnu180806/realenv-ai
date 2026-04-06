"""
MEDIUM TASK GRADER — Technical Outage Report
Difficulty: Medium (target score ~0.55–0.75)

The agent must:
1. Classify correctly as TECHNICAL / URGENT             (+0.20)
2. Request clarifying info (reproduction steps, env)    (+0.20)
3. Draft an acknowledgment response                     (+0.20)
4. Escalate to Engineering with a clear reason          (+0.40)

Escalation is REQUIRED here — resolving without escalation is penalised.
Partial credit if steps are done out of order.
"""

from __future__ import annotations
from typing import Tuple

from env.models import (
    ActionType, ClassifyAction, DraftResponseAction, ResolveAction,
    EscalateAction, RequestInfoAction, IssueCategory, Priority,
    EscalationTeam, EnvironmentState, AgentAction
)
from graders.base import BaseGrader


ESCALATION_KEYWORDS = {"engineering", "bug", "outage", "reproduce", "investigate", "technical"}
ACK_KEYWORDS = {"acknowledge", "aware", "investigating", "working on", "team", "escalate", "escalated"}
INFO_KEYWORDS = {"steps", "reproduce", "version", "browser", "os", "system", "error", "log", "screenshot"}


class TechnicalOutageGrader(BaseGrader):

    def grade_step(
        self, action: AgentAction, state: EnvironmentState
    ) -> Tuple[float, str, bool]:

        t = action.action_type

        if t == ActionType.CLASSIFY:
            action: ClassifyAction
            reward = 0.0
            msgs = []

            if action.category == IssueCategory.TECHNICAL:
                reward += 0.12
                msgs.append("✓ Correctly classified as technical.")
            else:
                msgs.append(f"✗ Wrong category — expected 'technical', got '{action.category}'.")

            if action.priority in (Priority.URGENT, Priority.HIGH):
                reward += 0.08
                msgs.append("✓ Urgent/high priority appropriate.")
            else:
                msgs.append("✗ Outage should be urgent or high priority.")

            return reward, " ".join(msgs), False

        if t == ActionType.REQUEST_INFO:
            action: RequestInfoAction
            body_lower = " ".join(action.questions).lower()
            hits = sum(1 for k in INFO_KEYWORDS if k in body_lower)
            reward = 0.0
            msgs = []

            if hits >= 3:
                reward = 0.20
                msgs.append("✓ Thorough info request covering key debug fields.")
            elif hits >= 1:
                reward = 0.10
                msgs.append("~ Partial info request — more debug context needed.")
            else:
                msgs.append("✗ Questions don't target technical debugging info.")

            return reward, " ".join(msgs), False

        if t == ActionType.DRAFT_RESPONSE:
            action: DraftResponseAction
            body_lower = action.body.lower()
            hits = sum(1 for k in ACK_KEYWORDS if k in body_lower)
            reward = 0.0
            msgs = []

            if hits >= 2:
                reward = 0.20
                msgs.append("✓ Good acknowledgment — customer feels heard.")
            elif hits == 1:
                reward = 0.10
                msgs.append("~ Acknowledgment is weak.")
            else:
                msgs.append("✗ Response doesn't acknowledge the outage.")

            return reward, " ".join(msgs), False

        if t == ActionType.ESCALATE:
            action: EscalateAction
            reward = 0.0
            msgs = []

            if action.team == EscalationTeam.ENGINEERING:
                reward += 0.25
                msgs.append("✓ Correctly escalated to Engineering.")
            elif action.team != EscalationTeam.NONE:
                reward += 0.10
                msgs.append(f"~ Escalated to '{action.team}' — Engineering preferred for outages.")
            else:
                msgs.append("✗ No team selected for escalation.")

            reason_lower = action.reason.lower()
            if any(k in reason_lower for k in ESCALATION_KEYWORDS):
                reward += 0.10
                msgs.append("✓ Escalation reason is technically specific.")
            else:
                msgs.append("~ Escalation reason lacks technical detail.")

            if action.internal_notes and len(action.internal_notes) > 20:
                reward += 0.05
                msgs.append("✓ Internal notes provided.")

            msgs.append("Ticket escalated. Episode complete.")
            return reward, " ".join(msgs), True

        if t == ActionType.RESOLVE:
            # Resolving a production outage without escalation is bad
            msgs = ["✗ A production outage should be escalated, not self-resolved."]
            return 0.0, " ".join(msgs), True

        if t == ActionType.REQUEST_INFO:
            return 0.05, "~ Requested info (may be useful).", False

        return 0.0, "Unknown action.", False

    def final_score(self, state: EnvironmentState) -> float:
        score = state.cumulative_reward
        # Full-pipeline bonus
        if state.classification_done and state.escalated:
            score = min(1.0, score + 0.05)
        return round(min(1.0, max(0.0, score)), 4)