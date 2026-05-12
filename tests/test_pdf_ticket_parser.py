"""Unit tests for CE ticket PDF parsing (no PDF binary required)."""

from __future__ import annotations

import pandas as pd

from DataLayer.pdf_ticket_parser import (
    _parse_block_to_record,
    clean_pdf_text,
    extract_field,
    parse_ticket_blocks,
)


SAMPLE_BLOCK = """
Title:
Student - Test User - testu (doors)
Requester:
Jane Doe
Opened by:
Jane Doe
Ticket Type:
New Employee
Number:
TK0000999
State:
Open
Employee Type:
Student
Employee Job Title:
Student Worker 1
Supervisor:
Jane Doe
Description:
Internal Notes:
Doors:
HCEB Test Door Group
Work Log(comments):
Awaiting provisioning confirmation.
"""


def test_extract_field_basic():
    assert extract_field(SAMPLE_BLOCK.strip(), "Ticket Type") == "New Employee"
    assert extract_field(SAMPLE_BLOCK.strip(), "Number") == "TK0000999"


def test_parse_block_to_record_core_fields():
    rec = _parse_block_to_record(SAMPLE_BLOCK.strip(), inferred_number=None)
    assert rec["parsed_successfully"] is True
    assert rec["Number"] == "TK0000999"
    assert rec["Requester"] == "Jane Doe"
    assert rec["Opened by"] == "Jane Doe"
    assert rec["Ticket Type"] == "New Employee"
    assert rec["Internal Notes"] is not None
    assert "HCEB Test Door Group" in rec["Internal Notes"]


def test_parse_ticket_blocks_with_preamble():
    cleaned = """
Title:
Alpha
Requester:
Beta
Ticket Type:
New Employee
Number:
TK0000999
State:
Closed
""".strip()
    blocks = parse_ticket_blocks(cleaned)
    assert len(blocks) == 1
    tid, body = blocks[0]
    assert tid == "TK0000999"
    assert "Title:" in body and "Alpha" in body
    assert "Requester:" in body


def test_clean_pdf_text_strips_report_banner():
    raw = """CE Tickets Details
Page 1
Run By : Someone
Title:
Hello
Number:
TK0000999
"""
    out = clean_pdf_text(raw)
    assert "CE Tickets Details" not in out
    assert "TK0000999" in out


def test_resolve_demo_identity_high_confidence():
    from DataLayer import pdf_ticket_parser as mod

    users = pd.DataFrame(
        {
            "SamAccountName": ["u_demo123"],
            "DisplayName": ["Demo User XX"],
        }
    )
    um = pd.DataFrame(
        {
            "real_sam_account": ["testu"],
            "fake_sam_account": ["u_demo123"],
        }
    )
    row = {
        "Title": "Student - Test User - testu (doors)",
        "Requester": None,
        "Opened by": None,
    }
    out = mod.resolve_demo_identity(
        row,
        demo_users=users,
        user_map=um,
        person_map=None,
    )
    assert out["MatchedDemoUser"] is True
    assert out["MatchConfidence"] == "high"
    assert out["DemoSamAccountName"] == "u_demo123"
