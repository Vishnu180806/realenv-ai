"""
baseline_agent.py — Optimized Baseline Agent (>0.9 Score Target)

Migrated to Groq with Chain-of-Thought reasoning and specialized 
logic for multi-issue enterprise tickets.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
import re
from typing import Any, Dict, List, Optional

try:
    from groq import Groq
except ImportError:
    print("Install groq: pip install groq")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
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

MODEL = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com")

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an ELITE customer support agent AI. Your goal is to achieve a perfect 1.0 score.

### STRATEGY & RULES:
1. **Analyze First**: For every turn, first write a brief "THOUGHT" block analyzing the ticket, identifying ALL sub-issues (Billing, Tech, Account), and checking the customer's tier.
2. **Multi-Issue Tickets**: If a ticket has multiple problems (e.g., Enterprise complaints), you MUST address ALL of them in your Request Info and Draft Response actions.
3. **Escalation**: 
   - Standard/Medium issues -> Engineering/Billing Team.
   - Enterprise/SLA/Critical issues -> MANAGEMENT. 
   - Always provide detailed "internal_notes" with a resolution plan (mentioning teams like "engineering", "billing", "account manager").
4. **Drafting**: Responses for Enterprise clients must be detailed (>200 chars), professional, and explicitly acknowledge every single issue raised.
5. **Request Info**: Ask specific questions for every category mentioned (e.g., ask for invoice details AND error logs AND user emails if all 3 are missing).
6. **Classify**: 
   - Billing is primary if money is involved. 
   - Use URGENT for Enterprise/Production Outages.

### OUTPUT FORMAT:
You must output your response in this exact format:
THOUGHT: <your reasoning here>
JSON: <valid JSON action object>

### JSON ACTIONS:
1. classify: {"action_type":"classify","category":"billing|technical|shipping|refund|account|other","priority":"low|medium|high|urgent","confidence":0.95}
2. draft_response: {"action_type":"draft_response","subject":"Re: ...","body":"Full response text...","tone":"professional"}
3. escalate: {"action_type":"escalate","team":"billing_team|engineering|logistics|management|none","reason":"Why escalating","internal_notes":"Notes for team"}
4. request_info: {"action_type":"request_info","questions":["Q1?","Q2?"],"body":"Customer message"}
5. resolve: {"action_type":"resolve","resolution_summary":"What was done","satisfied":true}
"""


def build_user_prompt(observation: Dict[str, Any]) -> str:
    ticket = observation["ticket"]
    history = observation["actions_taken"]
    
    # Extract latest feedback if available
    latest_feedback = ""
    if history:
        # The history usually contains dicts with 'feedback' if provided by the wrapper/runner
        # But our env.step returns it in 'info'. We'll assume the runner stores it.
        latest_action = history[-1]
        if "feedback" in latest_action:
            latest_feedback = f"LAST ACTION FEEDBACK: {latest_action['feedback']}\n"

    prompt = f"""{latest_feedback}TICKET:
Subject: {ticket['subject']}
Customer: {ticket['customer']['name']} (Tier: {ticket['customer']['account_tier']})
Body: {ticket['body']}

STATUS:
Step: {observation['step']} / {observation['max_steps']}
Classified: {observation['classification_done']}
Responded: {observation['response_drafted']}
Resolved: {observation['resolved']}
Escalated: {observation['escalated']}

PREVIOUS ACTIONS:
{json.dumps(history, indent=2) if history else 'None'}

PROVIDE YOUR THOUGHT AND JSON ACTION:"""
    return prompt


# ---------------------------------------------------------------------------
# Action parser
# ---------------------------------------------------------------------------

def parse_action(raw: str):
    """Parse LLM output containing Thought and JSON."""
    # Look for JSON block
    match = re.search(r'JSON:\s*(\{.*\})', raw, re.DOTALL)
    if not match:
        # Fallback: search for any { } block
        match = re.search(r'(\{.*\})', raw, re.DOTALL)
        
    if not match:
        raise ValueError(f"No JSON block found in response: {raw}")
        
    clean = match.group(1).strip()
    data = json.loads(clean)
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

def run_episode(task_id: str, client: Groq, verbose: bool = True) -> Dict:
    env = make_env(task_id)
    obs = env.reset()
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Task: {task_id} [{TASK_REGISTRY[task_id]['difficulty'].upper()}]")
        print(f"Ticket: {obs['ticket']['subject']}")
        print(f"{'='*60}")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    done = False
    step_log = []

    while not done:
        user_content = build_user_prompt(obs)
        messages.append({"role": "user", "content": user_content})

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=1000,
                temperature=0.0,
            )
            raw = response.choices[0].message.content
            
            # Print thought in verbose mode
            if verbose:
                thought_match = re.search(r'THOUGHT:\s*(.*?)(?=JSON:|$)', raw, re.DOTALL)
                if thought_match:
                    print(f"  Thought: {thought_match.group(1).strip()[:100]}...")

            messages.append({"role": "assistant", "content": raw})

            action = parse_action(raw)
            result = env.step(action)

            if verbose:
                print(f"  Step {obs['step']+1}: [{action.action_type}] → reward={result.reward:.3f} | {result.info['feedback'][:80]}")

            # Store feedback in the log so build_user_prompt can use it
            step_entry = {
                "step": obs["step"] + 1,
                "action_type": action.action_type,
                "reward": result.reward,
                "feedback": result.info["feedback"],
                "done": result.done,
            }
            step_log.append(step_entry)
            
            # Update history for build_user_prompt
            obs["actions_taken"].append(step_entry)

            obs = result.observation
            done = result.done

        except Exception as e:
            if verbose:
                print(f"  Step {obs['step']+1}: Error — {e}")
            step_log.append({"step": obs["step"]+1, "error": str(e), "reward": 0.0})
            try:
                env.step(ResolveAction(resolution_summary="Agent error", satisfied=False))
                done = True
            except:
                break

        time.sleep(0.2)

    final = env.final_score()
    if verbose:
        print(f"\n  FINAL SCORE: {final:.4f}")

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
    parser = argparse.ArgumentParser(description="Optimized Baseline Agent")
    parser.add_argument("--task", type=str, default=None, help="Single task ID to run")
    parser.add_argument("--output", type=str, default="results.json", help="Output JSON file")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not found.")
        sys.exit(1)

    client = Groq(api_key=api_key, base_url=API_BASE_URL)
    tasks_to_run = [args.task] if args.task else ALL_TASK_IDS

    results = []
    for task_id in tasks_to_run:
        result = run_episode(task_id, client, verbose=not args.quiet)
        results.append(result)

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    for r in results:
        bar = "█" * int(r["final_score"] * 20)
        print(f"  {r['task_id']:35s} [{r['difficulty']:6s}]  {r['final_score']:.4f}  {bar}")
    avg = sum(r["final_score"] for r in results) / len(results)
    print(f"\n  Average score: {avg:.4f}")
    print(f"{'='*60}")

    with open(args.output, "w") as f:
        json.dump({
            "model": MODEL,
            "results": results,
            "average_score": avg,
        }, f, indent=2)


if __name__ == "__main__":
    main()
