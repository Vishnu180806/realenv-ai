
 ---
title: RealEnv AI
emoji: 🤖
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---
 # CustomerSupportEnv

An OpenEnv-compliant real-world customer support resolution environment built for AI agent evaluation.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Interactive UI |
| GET | `/health` | Health check |
| GET | `/tasks` | List all tasks |
| GET | `/spec` | OpenEnv spec |
| GET | `/baseline` | Gold-standard baseline scores |
| POST | `/reset` | Start a new episode |
| POST | `/step` | Take an action |
| GET | `/state/{session_id}` | Get current state |
| GET | `/score/{session_id}` | Get final score |

## Tasks

- **billing_dispute_v1** — Easy: Billing dispute resolution
- **technical_outage_v1** — Medium: Technical outage escalation
- **enterprise_complaint_v1** — Hard: Enterprise multi-issue complaint

## Baseline Scores (Gold Standard Agent)

| Task | Difficulty | Score |
|------|-----------|-------|
| billing_dispute_v1 | Easy | 0.80 |
| technical_outage_v1 | Medium | 1.00 |
| enterprise_complaint_v1 | Hard | 0.95 |
| **Average** | | **0.9167** |
