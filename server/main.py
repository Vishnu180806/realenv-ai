"""
Typed models for the Customer Support Resolution Environment.
Follows OpenEnv spec: all state, actions, and observations are typed Pydantic models.
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import time


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class IssueCategory(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    SHIPPING = "shipping"
    REFUND = "refund"
    ACCOUNT = "account"
    OTHER = "other"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class ActionType(str, Enum):
    CLASSIFY = "classify"          # Categorise the issue
    DRAFT_RESPONSE = "draft_response"  # Write a customer-facing reply
    ESCALATE = "escalate"          # Escalate to a human team
    REQUEST_INFO = "request_info"  # Ask customer for more details
    RESOLVE = "resolve"            # Mark ticket as resolved


class EscalationTeam(str, Enum):
    BILLING_TEAM = "billing_team"
    ENGINEERING = "engineering"
    LOGISTICS = "logistics"
    MANAGEMENT = "management"
    NONE = "none"


# ---------------------------------------------------------------------------
# Ticket & Customer models
# ---------------------------------------------------------------------------

class CustomerProfile(BaseModel):
    customer_id: str
    name: str
    account_tier: str = "standard"   # standard / premium / enterprise
    previous_tickets: int = 0
    is_verified: bool = True


class Ticket(BaseModel):
    ticket_id: str
    subject: str
    body: str
    customer: CustomerProfile
    created_at: float = Field(default_factory=time.time)
    attachments: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Action models
# ---------------------------------------------------------------------------

class ClassifyAction(BaseModel):
    action_type: ActionType = ActionType.CLASSIFY
    category: IssueCategory
    priority: Priority
    confidence: float = Field(ge=0.0, le=1.0)


class DraftResponseAction(BaseModel):
    action_type: ActionType = ActionType.DRAFT_RESPONSE
    subject: str
    body: str
    tone: str = "professional"   # professional / empathetic / formal


class EscalateAction(BaseModel):
    action_type: ActionType = ActionType.ESCALATE
    team: EscalationTeam
    reason: str
    internal_notes: str = ""


class RequestInfoAction(BaseModel):
    action_type: ActionType = ActionType.REQUEST_INFO
    questions: List[str]
    body: str


class ResolveAction(BaseModel):
    action_type: ActionType = ActionType.RESOLVE
    resolution_summary: str
    satisfied: bool = True


AgentAction = ClassifyAction | DraftResponseAction | EscalateAction | RequestInfoAction | ResolveAction


# ---------------------------------------------------------------------------
# Observation & State models
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    observation: Dict[str, Any]
    reward: float = Field(ge=0.0, le=1.0)
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class EnvironmentState(BaseModel):
    task_id: str
    ticket: Ticket
    step_number: int = 0
    max_steps: int = 10
    actions_taken: List[Dict[str, Any]] = Field(default_factory=list)
    classification_done: bool = False
    response_drafted: bool = False
    resolved: bool = False
    escalated: bool = False
    cumulative_reward: float = 0.0
    done: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)