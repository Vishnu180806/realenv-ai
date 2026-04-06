"""
Task registry — maps task_id → {ticket_factory, grader, metadata}

Three tasks of increasing difficulty:
  easy:   billing_dispute_v1
  medium: technical_outage_v1
  hard:   enterprise_complaint_v1
"""

from __future__ import annotations
from env.models import Ticket, CustomerProfile
from graders.easy_grader import BillingDisputeGrader
from graders.medium_grader import TechnicalOutageGrader
from graders.hard_grader import EnterpriseComplaintGrader


# ---------------------------------------------------------------------------
# Ticket factories
# ---------------------------------------------------------------------------

def _billing_dispute_ticket() -> Ticket:
    return Ticket(
        ticket_id="TKT-001",
        subject="Charged twice for my subscription this month",
        body=(
            "Hello, I noticed that my credit card was charged twice for my "
            "monthly subscription on the 3rd and the 5th of this month. "
            "I only have one active subscription. The total duplicate charge "
            "is $49.99. I'd like a refund for the duplicate charge as soon as possible. "
            "My account email is john.smith@example.com. Order ID: ORD-88421."
        ),
        customer=CustomerProfile(
            customer_id="CUST-1001",
            name="John Smith",
            account_tier="standard",
            previous_tickets=1,
            is_verified=True,
        ),
        metadata={"order_id": "ORD-88421", "amount": 49.99, "currency": "USD"},
    )


def _technical_outage_ticket() -> Ticket:
    return Ticket(
        ticket_id="TKT-002",
        subject="URGENT: Production API completely down — affecting all customers",
        body=(
            "Our production integration with your API has been completely broken since "
            "2:00 AM UTC today. All API calls are returning 503 errors. This is affecting "
            "ALL of our end customers — approximately 2,400 users cannot access our product. "
            "We are losing ~$500/hour in revenue. We've already tried restarting our servers, "
            "refreshing API keys, and rolling back our deployment. Nothing works. "
            "This appears to be on your end. We need immediate escalation to your engineering team."
        ),
        customer=CustomerProfile(
            customer_id="CUST-2045",
            name="Sarah Chen",
            account_tier="premium",
            previous_tickets=3,
            is_verified=True,
        ),
        metadata={
            "affected_users": 2400,
            "revenue_impact_per_hour": 500,
            "error_code": "503",
            "reported_at": "2024-01-15T02:00:00Z",
        },
    )


def _enterprise_complaint_ticket() -> Ticket:
    return Ticket(
        ticket_id="TKT-003",
        subject="CRITICAL: Multiple issues — Invoice error + API broken + 2 users locked out",
        body=(
            "We are an enterprise customer (Contract #ENT-7821) and we have THREE critical issues "
            "that need immediate resolution:\n\n"
            "1. BILLING: Our January invoice (#INV-2024-01) shows a charge of $12,450 but our "
            "contract rate is $9,800/month. We've been overcharged by $2,650.\n\n"
            "2. TECHNICAL: Our Webhook integration has been failing since the platform update on "
            "Jan 12. Our engineering team is getting 401 Unauthorized errors on all POST requests "
            "to /api/v2/webhooks even with valid OAuth tokens.\n\n"
            "3. ACCOUNT: Two of our admin users (alice@corp.com and bob@corp.com) have had their "
            "accounts revoked without any notice. They cannot log in. This is blocking our "
            "compliance reporting due tomorrow.\n\n"
            "We expect a response within 2 hours per our SLA. If unresolved we will initiate "
            "contract termination proceedings."
        ),
        customer=CustomerProfile(
            customer_id="CUST-ENT-0042",
            name="Marcus Webb (VP Operations, TechCorp Inc.)",
            account_tier="enterprise",
            previous_tickets=12,
            is_verified=True,
        ),
        attachments=["invoice_Jan2024.pdf", "api_error_logs.txt", "account_screenshot.png"],
        metadata={
            "contract_id": "ENT-7821",
            "monthly_value": 9800,
            "overcharge": 2650,
            "sla_hours": 2,
            "locked_users": ["alice@corp.com", "bob@corp.com"],
        },
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASK_REGISTRY = {
    "billing_dispute_v1": {
        "ticket_factory": _billing_dispute_ticket,
        "grader": BillingDisputeGrader(),
        "difficulty": "easy",
        "description": "Simple billing dispute — classify, respond, resolve.",
        "max_steps": 8,
    },
    "technical_outage_v1": {
        "ticket_factory": _technical_outage_ticket,
        "grader": TechnicalOutageGrader(),
        "difficulty": "medium",
        "description": "Production API outage — classify, gather info, acknowledge, escalate.",
        "max_steps": 10,
    },
    "enterprise_complaint_v1": {
        "ticket_factory": _enterprise_complaint_ticket,
        "grader": EnterpriseComplaintGrader(),
        "difficulty": "hard",
        "description": "Enterprise multi-issue complaint — all four pipeline steps required.",
        "max_steps": 12,
    },
}

ALL_TASK_IDS = list(TASK_REGISTRY.keys())
