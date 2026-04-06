from __future__ import annotations
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ActionType(str, Enum):
    CLASSIFY = "classify"
    DRAFT_RESPONSE = "draft_response"
    ESCALATE = "escalate"
    REQUEST_INFO = "request_info"
    RESOLVE = "resolve"

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

class EscalationTeam(str, Enum):
    BILLING_TEAM = "billing_team"
    ENGINEERING = "engineering"
    LOGISTICS = "logistics"
    MANAGEMENT = "management"
    NONE = "none"

class AgentAction(BaseModel):
    action_type: ActionType

class ClassifyAction(AgentAction):
    action_type: ActionType = ActionType.CLASSIFY
    category: IssueCategory
    priority: Priority
    confidence: float

class DraftResponseAction(AgentAction):
    action_type: ActionType = ActionType.DRAFT_RESPONSE
    subject: str
    body: str
    tone: str = "professional"

class EscalateAction(AgentAction):
    action_type: ActionType = ActionType.ESCALATE
    team: EscalationTeam
    reason: str
    internal_notes: Optional[str] = None

class RequestInfoAction(AgentAction):
    action_type: ActionType = ActionType.REQUEST_INFO
    questions: List[str]
    body: str

class ResolveAction(AgentAction):
    action_type: ActionType = ActionType.RESOLVE
    resolution_summary: str
    satisfied: bool

class CustomerProfile(BaseModel):
    customer_id: str
    name: str
    account_tier: str
    previous_tickets: int = 0
    is_verified: bool = False

class Ticket(BaseModel):
    ticket_id: str
    subject: str
    body: str
    customer: CustomerProfile
    attachments: List[str] = []
    metadata: Dict[str, Any] = {}

class EnvironmentState(BaseModel):
    ticket: Ticket
    step: int = 0
    max_steps: int = 10
    actions_taken: List[Dict[str, Any]] = []
    cumulative_reward: float = 0.0
    classification_done: bool = False
    response_drafted: bool = False
    resolved: bool = False
    escalated: bool = False
