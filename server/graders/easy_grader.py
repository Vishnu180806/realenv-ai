from __future__ import annotations
from typing import Tuple
from env.models import (
    ActionType, ClassifyAction, ResolveAction, IssueCategory, 
    Priority, EnvironmentState, AgentAction
)
from graders.base import BaseGrader

class BillingDisputeGrader(BaseGrader):
    def grade_step(
        self, action: AgentAction, state: EnvironmentState
    ) -> Tuple[float, str, bool]:
        t = action.action_type

        if t == ActionType.CLASSIFY:
            action: ClassifyAction
            reward = 0.0
            msgs = []
            if action.category == IssueCategory.BILLING:
                reward += 0.2
                msgs.append("✓ Correctly classified as billing.")
            else:
                msgs.append(f"✗ Wrong category: {action.category}")
            
            if action.priority == Priority.MEDIUM:
                reward += 0.1
                msgs.append("✓ Priority okay.")
            
            return reward, " ".join(msgs), False

        if t == ActionType.RESOLVE:
            action: ResolveAction
            if action.satisfied:
                return 0.5, "✓ Ticket resolved satisfactorily.", True
            return 0.1, "~ Partailly resolved.", True

        return 0.0, "Action noted.", False

    def final_score(self, state: EnvironmentState) -> float:
        # Platform requires score strictly between 0 and 1.
        score = min(1.0, state.cumulative_reward)
        return round(min(0.99, max(0.01, score)), 4)
