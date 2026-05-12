# AccessGraph

AccessGraph is an AI-assisted access governance system for onboarding and helpdesk workflows.

## Goal
Recommend access rights for new employees or access requests using:
- role-based patterns
- similar-user evidence
- risk scoring
- explainable approval notes

## Architecture
- DataLayer: load and sanitize employee/access data
- DeterministicLayer: RBAC confidence and permission rules
- MLLayer: similar-user matching
- ProductLayer: final recommendation, risk, and explanation
- UI: Streamlit demo

## Current Hackathon Priority
1. Sanitized dataset
2. RBAC permission recommendations
3. Similar-user evidence
4. Risk flags
5. Streamlit demo
6. Approval note generator

## Rules
- Do not expose real names, NetIDs, emails, or sensitive data.
- Make small focused edits.
- Preserve modular structure.
- Prefer working demo over perfect architecture.
- Every recommendation should explain why.
- 
## Editing Rules

- Never modify files automatically.
- Always explain proposed changes first.
- Show unified diffs before edits.
- Wait for explicit approval before applying changes.
- Prefer additive localized changes.
- Do not rewrite unrelated files.
## Canonical Entry Points

Main recommendation engine:
- ProductLayer/AccessRecommendationEngine.py

Main UI:
- ProductLayer/app.py

Primary dataset:
- data/processed/clean_users.parquet

Reference data:
- data/raw/full_time_employee_access.xlsx
- data/raw/student_employee_access.xlsx

Validation:
- scripts/validate_engine_ambiguity.py
- evaluation/backtest.py

## Known Caveats

- Current parquet contains many zero-group users which weakens ML similarity.
- Backtest currently hides random groups and is not fully time-safe.
- Student reference templates may still collide across supervisors.
- Reference ambiguity logic is intentionally conservative.
- Streamlit caching may require restart after pipeline changes.