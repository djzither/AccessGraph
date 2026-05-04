AccessGraph
AccessGraph is a hybrid access recommendation engine that analyzes user permissions and suggests appropriate access for new or existing employees. It combines deterministic role-based rules with machine learning to improve consistency and reduce over-permissioning.
Overview
Managing access across systems such as Active Directory, ServiceNow, and internal tools is often manual and inconsistent. AccessGraph helps automate this process by:
Learning access patterns from existing users
Recommending permissions based on role and department
Identifying unusual or high-risk access
Supporting onboarding and access audits
Architecture
AccessGraph is built as a multi-layer pipeline:
Data Layer
Loads and cleans raw access data (e.g., AD exports)
Converts group strings into structured lists
Outputs a processed dataset for downstream use
Deterministic Layer
Builds access patterns using Title and Department
Computes confidence scores for each permission
Classifies access patterns (Baseline, Common, Rare, Unique)
Machine Learning Layer
Constructs a user-permission matrix
Uses cosine similarity to identify similar users
Generates recommendations based on nearest neighbors
Product (Hybrid) Layer
Combines deterministic and ML outputs
Produces final decisions:
Auto Approve
Manual Review
Low Confidence
Flags sensitive or risky permissions
Features
Hybrid recommendation system (rules + ML)
Role-based access pattern detection
Supervisor and outlier handling
Integration with reference access sheets
Extensible design for API integration and automation
Tech Stack
Python (Pandas, scikit-learn)
Cosine similarity for user matching
Parquet for efficient data storage
Usage
Place raw data in:
data/raw/
Run data cleaning:
python scripts/run_real_pipeline.py
Generate recommendations:
python scripts/run_combined_pipeline.py
Example Use Case
Given a new hire with a known role and department, AccessGraph:
Identifies similar users
Extracts common permissions
Filters out risky or irrelevant access
Outputs a ranked list of recommended permissions
Future Work
Integration with ServiceNow and Active Directory APIs
Automated provisioning workflows
Feedback loop for improving recommendations
LLM-assisted reasoning for ambiguous cases
