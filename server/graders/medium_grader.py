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

    def grade_step(
        self, action: AgentAction, state: EnvironmentState
    ) -> Tuple[float, str, bool]:

        t = action.action_type

        if t == ActionType.CLASSIFY:
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

            return max(0.01, reward), " ".join(msgs), False

        if t == ActionType.REQUEST_INFO:
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

            return max(0.01, reward), " ".join(msgs), False

        if t == ActionType.DRAFT_RESPONSE:
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

            return max(0.01, reward), " ".join(msgs), False

        if t == ActionType.ESCALATE:
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
            return max(0.01, reward), " ".join(msgs), True

        if t == ActionType.RESOLVE:
            # Resolving a production outage without escalation is bad
            msgs = ["✗ A production outage should be escalated, not self-resolved."]
            return 0.01, " ".join(msgs), True

        return 0.01, "Unknown action.", False

    def final_score(self, state: EnvironmentState) -> float:
        """
        Return final episode score strictly between 0 and 1.
        Platform validation requires: 0 < score < 1 (not 0.0, not 1.0)
        """
        score = state.cumulative_reward
        
        # Full-pipeline bonus (but keep score under 0.99)
        if state.classification_done and state.escalated:
            score = min(0.94, score + 0.05)
        
        # Clamp strictly within (0.01, 0.99)
        score = max(0.01, min(0.99, score))
        
        return round(score, 4)