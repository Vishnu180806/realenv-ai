"""
mock_agent.py — Simple random agent for CustomerSupportEnv
Demonstrates the environment without requiring an LLM API key.
"""

from __future__ import annotations
import random
import time
from env.environment import make_env
from env.models import (
    ClassifyAction, DraftResponseAction, EscalateAction,
    RequestInfoAction, ResolveAction, IssueCategory, Priority,
    EscalationTeam, ActionType
)
from tasks.registry import ALL_TASK_IDS

def run_mock_episode(task_id: str):
    env = make_env(task_id)
    obs = env.reset()
    
    print(f"\n{'='*60}")
    print(f"MOCK RUN: {task_id}")
    print(f"Ticket: {obs['ticket']['subject']}")
    print(f"{'='*60}")

    done = False
    while not done:
        # Simple heuristic for random actions
        if not obs['classification_done']:
            action = ClassifyAction(
                category=random.choice(list(IssueCategory)),
                priority=random.choice(list(Priority)),
                confidence=0.9
            )
        elif not obs['response_drafted']:
            action = DraftResponseAction(
                subject=f"Re: {obs['ticket']['subject']}",
                body="We have received your request and are looking into it.",
                tone="professional"
            )
        else:
            # Decide to escalate or resolve randomly
            if random.random() > 0.5:
                action = EscalateAction(
                    team=random.choice([EscalationTeam.ENGINEERING, EscalationTeam.BILLING_TEAM]),
                    reason="Technical complexity requires specialist review.",
                    internal_notes="Mock escalation for testing."
                )
            else:
                action = ResolveAction(
                    resolution_summary="Resolved via mock automated script.",
                    satisfied=True
                )

        result = env.step(action)
        print(f"  Step {obs['step']+1}: [{action.action_type}] → reward={result.reward:.2f} | {result.info['feedback']}")
        
        obs = result.observation
        done = result.done
        time.sleep(0.5)

    print(f"\n  FINAL SCORE: {env.final_score():.4f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    task = random.choice(ALL_TASK_IDS)
    run_mock_episode(task)
