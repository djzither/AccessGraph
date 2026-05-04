AccessGraph

AccessGraph is a hybrid access recommendation engine designed to automate and improve role-based access provisioning. It analyzes historical user permissions and organizational patterns to recommend appropriate access for new or existing employees.

Overview

Provisioning access across systems is often manual, inconsistent, and error-prone. AccessGraph addresses this by combining:

Deterministic rules based on role and department
Machine learning similarity between users
Reference access sheets for validation
Risk-aware filtering for sensitive permissions

The result is a structured, explainable recommendation system for access control.

Features
Role-Based Recommendations
Suggests permissions using patterns from users with the same title and department.
ML Similarity Engine
Uses cosine similarity on user-permission matrices to find comparable users.
Hybrid Decision Layer
Combines rule-based and ML outputs into a final recommendation with confidence scores.
Risk Filtering
Flags sensitive or high-risk permissions for manual review.
Access Pattern Analysis
Classifies permissions as baseline, common, rare, or unique.
Reference Sheet Matching
Aligns recommendations with official access documentation when available.
Data Pipeline
Ingestion
Loads raw Active Directory exports (CSV/XLSX)
Cleaning
Parses group memberships
Removes invalid or missing entries
Outputs structured data (GroupsList)
Processing
Builds role-based permission matrices
Computes confidence scores per permission
Recommendation Engine
Deterministic rules (role + department)
ML similarity (nearest users)
Hybrid merge with final scoring
Project Structure
AccessGraph/
│
├── data/
│   ├── raw/              # Raw AD exports (ignored by git)
│   ├── processed/        # Cleaned parquet data
│
├── DataLayer/
│   └── cleaner.py        # Data cleaning logic
│
├── ModelLayer/
│   └── similarity_model.py
│
├── ProductLayer/
│   ├── rules_recommender.py
│   ├── ml_recommender.py
│   ├── hybrid_recommender.py
│
├── scripts/
│   ├── run_real_pipeline.py
│   ├── run_combined_pipeline.py
│
└── README.md
Example Output

Each recommendation includes:

GroupName
FinalScore
FinalDecision (Auto Approve / Manual Review / Low Confidence)
Confidence (rule-based or ML-based)
Supporting evidence (similar users, counts)
Use Cases
New employee onboarding
Access auditing and cleanup
Identifying over-permissioned users
Standardizing access across teams
Future Improvements
Integration with ServiceNow APIs
Real-time provisioning workflows
LLM-assisted decision explanations
Feedback loop for continuous learning
Setup
Install dependencies
pip install -r requirements.txt
Place raw AD export in:
data/raw/
Run pipeline:
python scripts/run_real_pipeline.py
Generate recommendations:
python scripts/run_combined_pipeline.py
