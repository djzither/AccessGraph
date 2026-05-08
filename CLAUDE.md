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