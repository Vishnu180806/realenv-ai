"""
baseline_agent.py — Baseline inference script

Runs a simple LLM-based agent against all three tasks and reports
reproducible scores. Uses the Anthropic API (claude-sonnet-4-20250514).

Usage:
    python baseline_agent.py                    # run all tasks
    python baseline_agent.py --task billing_dispute_v1
    python baseline_agent.py --output results.json
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

try:
    import anthropic
except ImportError:
    print("Install anthropic: pip install anthropic")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.environment import make_env
from env.models import (
    ClassifyAction, DraftResponseAction, EscalateAction,
    RequestInfoAction, ResolveAction, ActionType
)
from tasks.registry import ALL_TASK_IDS, TASK_REGISTRY


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert customer support agent AI. 
You are given a support ticket and must handle it step by step.

At each step, output ONLY a valid JSON object representing your next action.
Choose from these action types:

1. classify: {"action_type":"classify","category":"billing|technical|shipping|refund|account|other","priority":"low|medium|high|urgent","confidence":0.95}
2. draft_response: {"action_type":"draft_response","subject":"Re: ...","body":"Full response text...","tone":"professional"}
3. escalate: {"action_type":"escalate","team":"billing_team|engineering|logistics|management|none","reason":"Why escalating","internal_notes":"Notes for team"}
4. request_info: {"action_type":"request_info","questions":["Q1?","Q2?"],"body":"Customer-facing message asking for info"}
5. resolve: {"action_type":"resolve","resolution_summary":"What was done","satisfied":true}

Output ONLY the JSON, no other text."""


def build_user_prompt(observation: Dict[str, Any]) -> str:
    ticket = observation["ticket"]
    history = observation["actions_taken"]
    
    prompt = f"""TICKET:
Subject: {ticket['subject']}
Customer: {ticket['customer']['name']} (Tier: {ticket['customer']['account_tier']})
Body: {ticket['body']}

STEP: {observation['step']} / {observation['max_steps']}
Classification done: {observation['classification_done']}
Response drafted: {observation['response_drafted']}
Resolved: {observation['resolved']}
Escalated: {observation['escalated']}

PREVIOUS ACTIONS: {json.dumps(history, indent=2) if history else 'None'}

What is your next action? Output ONLY valid JSON."""
    return prompt


# ---------------------------------------------------------------------------
# Action parser
# ---------------------------------------------------------------------------

def parse_action(raw: str):
    """Parse LLM output into a typed action object."""
    # Strip markdown fences if present
    clean = raw.strip()
    if clean.startswith("```"):
        lines = clean.split("\n")
        clean = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
    
    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        # Try to find JSON block
        import re
        match = re.search(r'\{.*\}', clean, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            raise
            
    t = data.get("action_type")
    
    if t == ActionType.CLASSIFY:
        return ClassifyAction(**data)
    elif t == ActionType.DRAFT_RESPONSE:
        return DraftResponseAction(**data)
    elif t == ActionType.ESCALATE:
        return EscalateAction(**data)
    elif t == ActionType.REQUEST_INFO:
        return RequestInfoAction(**data)
    elif t == ActionType.RESOLVE:
        return ResolveAction(**data)
    else:
        raise ValueError(f"Unknown action_type: {t}")


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def run_episode(task_id: str, client: anthropic.Anthropic, verbose: bool = True) -> Dict:
    env = make_env(task_id)
    obs = env.reset()
    
    if verbose:
        print(f"\\n{'='*60}")
        print(f"Task: {task_id} [{TASK_REGISTRY[task_id]['difficulty'].upper()}]")
        print(f"Ticket: {obs['ticket']['subject']}")
        print(f"{'='*60}")

    messages = []
    done = False
    step_log = []

    while not done:
        user_content = build_user_prompt(obs)
        messages.append({"role": "user", "content": user_content})

        try:
            response = client.messages.create(
                model="claude-3-sonnet-20240229", # Updated to a valid model name
                max_tokens=600,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
            raw = response.content[0].text
            messages.append({"role": "assistant", "content": raw})

            action = parse_action(raw)
            result = env.step(action)

            if verbose:
                print(f"  Step {obs['step']+1}: [{action.action_type}] → reward={result.reward:.3f} | {result.info['feedback'][:80]}")

            step_log.append({
                "step": obs["step"] + 1,
                "action_type": action.action_type,
                "reward": result.reward,
                "feedback": result.info["feedback"],
                "done": result.done,
            })

            obs = result.observation
            done = result.done

        except Exception as e:
            if verbose:
                print(f"  Step {obs['step']+1}: Error — {e}")
            step_log.append({"step": obs["step"]+1, "error": str(e), "reward": 0.0})
            break

        time.sleep(0.3)

    final = env.final_score()
    if verbose:
        print(f"\\n  FINAL SCORE: {final:.4f}")

    return {
        "task_id": task_id,
        "difficulty": TASK_REGISTRY[task_id]["difficulty"],
        "final_score": final,
        "steps": step_log,
        "total_steps": len(step_log),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Baseline agent for CustomerSupportEnv")
    parser.add_argument("--task", type=str, default=None, help="Single task ID to run")
    parser.add_argument("--output", type=str, default="results.json", help="Output JSON file")
    parser.add_argument("--quiet", action="store_true", help="Suppress step-by-step output")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    tasks_to_run = [args.task] if args.task else ALL_TASK_IDS

    results = []
    for task_id in tasks_to_run:
        result = run_episode(task_id, client, verbose=not args.quiet)
        results.append(result)

    # Summary
    print(f"\\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    for r in results:
        bar = "█" * int(r["final_score"] * 20)
        print(f"  {r['task_id']:35s} [{r['difficulty']:6s}]  {r['final_score']:.4f}  {bar}")
    avg = sum(r["final_score"] for r in results) / len(results)
    print(f"\\n  Average score: {avg:.4f}")
    print(f"{'='*60}")

    with open(args.output, "w") as f:
        json.dump({
            "model": "claude-3-sonnet-20240229",
            "results": results,
            "average_score": avg,
        }, f, indent=2)
    print(f"\\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
