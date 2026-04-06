
"""
app.py — FastAPI server for CustomerSupportEnv
Deployed on Hugging Face Spaces via Dockerfile.

Endpoints:
  GET  /                   — Interactive 3D UI (Three.js background)
  POST /reset              — Start a new episode
  POST /step               — Take an action
  GET  /state              — Get current state
  GET  /score              — Get final score
  GET  /tasks              — List all tasks
  GET  /spec               — Return openenv.yaml spec
  GET  /health             — Health check
"""

from __future__ import annotations
import json
import os
import sys
import yaml
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.environment import make_env, CustomerSupportEnv
from env.models import (
    ClassifyAction, DraftResponseAction, EscalateAction,
    RequestInfoAction, ResolveAction, ActionType
)
from tasks.registry import ALL_TASK_IDS, TASK_REGISTRY


app = FastAPI(
    title="CustomerSupportEnv",
    description="OpenEnv-compliant real-world customer support resolution environment.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# In-memory session store (single session for Spaces demo)
_env_store: Dict[str, CustomerSupportEnv] = {}


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str


class ActionRequest(BaseModel):
    session_id: str
    action: Dict[str, Any]


class ScoreRequest(BaseModel):
    session_id: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_action(data: Dict[str, Any]):
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


import uuid

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": tid,
                "difficulty": TASK_REGISTRY[tid]["difficulty"],
                "description": TASK_REGISTRY[tid]["description"],
                "max_steps": TASK_REGISTRY[tid]["max_steps"],
            }
            for tid in ALL_TASK_IDS
        ]
    }


@app.get("/spec")
def get_spec():
    spec_path = os.path.join(os.path.dirname(__file__), "openenv.yaml")
    with open(spec_path) as f:
        return yaml.safe_load(f)


@app.post("/reset")
def reset(req: ResetRequest):
    if req.task_id not in TASK_REGISTRY:
        raise HTTPException(400, f"Unknown task_id '{req.task_id}'. Options: {ALL_TASK_IDS}")
    
    session_id = str(uuid.uuid4())
    env = make_env(req.task_id)
    obs = env.reset()
    _env_store[session_id] = env
    
    return {"session_id": session_id, "observation": obs}


@app.post("/step")
def step(req: ActionRequest):
    env = _env_store.get(req.session_id)
    if env is None:
        raise HTTPException(404, f"Session '{req.session_id}' not found. Call /reset first.")
    
    try:
        action = _parse_action(req.action)
    except (ValueError, Exception) as e:
        raise HTTPException(422, f"Invalid action: {e}")
    
    try:
        result = env.step(action)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    
    return {
        "observation": result.observation,
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
    }


@app.get("/state/{session_id}")
def get_state(session_id: str):
    env = _env_store.get(session_id)
    if env is None:
        raise HTTPException(404, f"Session '{session_id}' not found.")
    return json.loads(env.to_json())


@app.get("/score/{session_id}")
def get_score(session_id: str):
    env = _env_store.get(session_id)
    if env is None:
        raise HTTPException(404, f"Session '{session_id}' not found.")
    return {"session_id": session_id, "final_score": env.final_score()}


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    _env_store.pop(session_id, None)
    return {"deleted": session_id}


@app.get("/baseline")
def get_baseline():
    """Return pre-computed gold-standard baseline benchmark results."""
    results_path = os.path.join(os.path.dirname(__file__), "results.json")
    if not os.path.exists(results_path):
        raise HTTPException(404, "No baseline results found. Run benchmark.py first.")
    with open(results_path) as f:
        return json.load(f)


@app.get("/", response_class=FileResponse)
def frontend():
    """Serve the interactive Nebula UI."""
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    return FileResponse(index_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
