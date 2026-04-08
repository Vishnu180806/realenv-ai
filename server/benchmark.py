"""
benchmark.py — Deterministic baseline benchmark (NO API key needed)

Runs gold-standard optimal action sequences through each task's grader
to produce reproducible baseline scores.

Usage:
    python benchmark.py                     # run all tasks
    python benchmark.py --task billing_dispute_v1
    python benchmark.py --output results.json
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.environment import make_env
from env.models import (
    ClassifyAction, DraftResponseAction, EscalateAction,
    RequestInfoAction, ResolveAction, ActionType,
    IssueCategory, Priority, EscalationTeam
)
from tasks.registry import ALL_TASK_IDS, TASK_REGISTRY


# ---------------------------------------------------------------------------
# Gold-standard action sequences (optimal play for each task)
# ---------------------------------------------------------------------------

GOLD_ACTIONS = {
    "task_1": [
        ClassifyAction(
            category=IssueCategory.BILLING,
            priority=Priority.MEDIUM,
            confidence=0.95,
        ),
        DraftResponseAction(
            subject="Re: Charged twice for my subscription this month",
            body=(
                "Dear John,\n\n"
                "Thank you for reaching out. I've reviewed your account and can confirm "
                "that a duplicate charge of $49.99 was processed on your credit card. "
                "I sincerely apologize for the billing error.\n\n"
                "I have initiated a full refund for the duplicate charge (Order ID: ORD-88421). "
                "You should see the refund reflected on your statement within 3-5 business days.\n\n"
                "If you have any further questions, please don't hesitate to reach out.\n\n"
                "Best regards,\nSupport Team"
            ),
        ),
        ResolveAction(
            resolution_summary="Duplicate billing charge of $49.99 confirmed and refunded. "
                               "Customer notified of 3-5 day refund timeline.",
            satisfied=True,
        ),
    ],

    "task_2": [
        ClassifyAction(
            category=IssueCategory.TECHNICAL,
            priority=Priority.URGENT,
            confidence=0.98,
        ),
        RequestInfoAction(
            questions=[
                "Can you provide the exact error messages or HTTP status codes you're seeing?",
                "What steps have you taken to reproduce the issue?",
                "Which API version and endpoints are affected?",
                "Can you share relevant error logs or screenshots?",
                "What is your system/browser/OS environment?",
            ],
            body="We need additional technical details to investigate this outage.",
        ),
        DraftResponseAction(
            subject="Re: URGENT: Production API completely down",
            body=(
                "Dear Sarah,\n\n"
                "Thank you for reporting this critical issue. We acknowledge that your "
                "production API integration is experiencing 503 errors and understand "
                "the severe impact on your 2,400 users.\n\n"
                "Our team is actively aware of this situation and we are investigating "
                "the root cause immediately. We are escalating this to our engineering "
                "team for priority resolution.\n\n"
                "We will provide an update within the next 30 minutes. Your account "
                "has been flagged as a premium priority case.\n\n"
                "Best regards,\nSupport Team"
            ),
        ),
        EscalateAction(
            team=EscalationTeam.ENGINEERING,
            reason=(
                "Production API outage affecting 2,400 users with 503 errors. "
                "Engineering team needs to investigate server-side root cause. "
                "Revenue impact of $500/hour. Customer has already ruled out "
                "client-side issues (restarted servers, refreshed API keys, "
                "rolled back deployment)."
            ),
            internal_notes=(
                "Premium customer (CUST-2045) experiencing total API outage since "
                "02:00 UTC. All endpoints returning 503. Customer-side troubleshooting "
                "exhausted. Likely infrastructure or deployment issue on our end. "
                "Engineering to investigate immediately — SLA at risk."
            ),
        ),
    ],

    "task_3": [
        ClassifyAction(
            category=IssueCategory.BILLING,
            priority=Priority.URGENT,
            confidence=0.92,
        ),
        RequestInfoAction(
            questions=[
                "Can you provide the exact invoice line items showing the overcharge on billing?",
                "What OAuth token scopes and API endpoint are returning the 401 authentication error?",
                "When were the user accounts for alice@corp.com and bob@corp.com last accessible for login?",
                "Can you share the error log screenshots from the webhook integration failure?",
                "What is your current system version and browser environment?",
            ],
            body="We need details on all three issues to investigate properly.",
        ),
        DraftResponseAction(
            subject="Re: CRITICAL: Multiple issues — Invoice error + API broken + 2 users locked out",
            body=(
                "Dear Marcus,\n\n"
                "Thank you for bringing these critical issues to our attention. As a valued "
                "enterprise customer, your concerns are our top priority. I want to address "
                "each of your three issues:\n\n"
                "1. BILLING/INVOICE: I've flagged Invoice #INV-2024-01 for immediate review. "
                "The $2,650 overcharge against your contract rate of $9,800/month will be "
                "investigated by our billing team and corrected within 24 hours.\n\n"
                "2. TECHNICAL/API: Our engineering team is investigating the 401 Unauthorized "
                "errors on your webhook integration. The authentication token issue following "
                "our Jan 12 platform update is being treated as a priority bug.\n\n"
                "3. ACCOUNT ACCESS: We are working to restore access for alice@corp.com and "
                "bob@corp.com immediately. Our account manager will ensure the user permission "
                "revocations are reversed so your compliance reporting is not blocked.\n\n"
                "Per your SLA (Contract #ENT-7821), we will provide a comprehensive update "
                "within 2 hours. This has been escalated to senior management.\n\n"
                "Best regards,\nSupport Team"
            ),
        ),
        EscalateAction(
            team=EscalationTeam.MANAGEMENT,
            reason=(
                "Enterprise customer (Contract #ENT-7821) with three simultaneous critical "
                "issues: billing overcharge, API authentication failure, and account lockouts. "
                "SLA deadline approaching. Management escalation required for enterprise accounts."
            ),
            internal_notes=(
                "High-value enterprise customer threatening contract termination. "
                "Three issues requiring cross-team coordination:\n"
                "- Billing team: Review and correct invoice overcharge of $2,650\n"
                "- Engineering: Investigate 401 errors on /api/v2/webhooks post-update\n"
                "- Account manager: Restore access for 2 locked admin users\n"
                "Priority: Respond within 2 hours per SLA. Management to coordinate "
                "all three teams for 24 hours resolution target."
            ),
        ),
    ],
}


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(task_id: str, verbose: bool = True) -> dict:
    env = make_env(task_id)
    obs = env.reset()
    actions = GOLD_ACTIONS[task_id]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Task: {task_id} [{TASK_REGISTRY[task_id]['difficulty'].upper()}]")
        print(f"Ticket: {obs['ticket']['subject']}")
        print(f"{'='*60}")

    step_log = []
    done = False

    for action in actions:
        if done:
            break

        result = env.step(action)

        if verbose:
            fb = result.info['feedback']
            print(f"  Step {obs['step']+1}: [{action.action_type}]"
                  f" → reward={result.reward:.3f} | {fb[:90]}")

        step_log.append({
            "step": obs["step"] + 1,
            "action_type": action.action_type,
            "reward": round(result.reward, 4),
            "feedback": result.info["feedback"],
            "done": result.done,
        })

        obs = result.observation
        done = result.done

    final = env.final_score()
    if verbose:
        print(f"\n  ★ FINAL SCORE: {final:.4f}")

    return {
        "task_id": task_id,
        "difficulty": TASK_REGISTRY[task_id]["difficulty"],
        "final_score": round(final, 4),
        "steps": step_log,
        "total_steps": len(step_log),
    }


def main():
    parser = argparse.ArgumentParser(description="Deterministic baseline benchmark")
    parser.add_argument("--task", type=str, default=None, help="Single task ID")
    parser.add_argument("--output", type=str, default="results.json", help="Output JSON")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    tasks = [args.task] if args.task else ALL_TASK_IDS
    results = []

    for tid in tasks:
        r = run_benchmark(tid, verbose=not args.quiet)
        results.append(r)

    # Summary
    print(f"\n{'='*60}")
    print("BASELINE BENCHMARK RESULTS (Gold-Standard Optimal Play)")
    print(f"{'='*60}")
    for r in results:
        bar = "█" * int(r["final_score"] * 20)
        spaces = " " * (20 - len(bar))
        print(f"  {r['task_id']:35s} [{r['difficulty']:6s}]  "
              f"{r['final_score']:.4f}  {bar}{spaces}")
    avg = sum(r["final_score"] for r in results) / len(results)
    print(f"\n  Average score: {avg:.4f}")
    print(f"{'='*60}")

    output_path = os.path.join(os.path.dirname(__file__), args.output)
    with open(output_path, "w") as f:
        json.dump({
            "agent": "gold_standard_baseline",
            "description": "Deterministic optimal actions — upper bound baseline",
            "results": results,
            "average_score": round(avg, 4),
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
