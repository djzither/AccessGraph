# AccessGraph

AccessGraph is a hybrid access recommendation engine designed to automate and improve role-based access provisioning. It analyzes historical user permissions and organizational patterns to recommend appropriate access for new or existing employees.

## Overview

Provisioning access across systems is often manual, inconsistent, and error-prone. AccessGraph addresses this by combining:

- Deterministic rules based on role and department  
- Machine learning similarity between users  
- Reference access sheets for validation  
- Risk-aware filtering for sensitive permissions  

The result is a structured, explainable recommendation system for access control.

## Features

- **Role-Based Recommendations**  
  Suggests permissions using patterns from users with the same title and department  

- **ML Similarity Engine**  
  Uses cosine similarity on user-permission matrices  

- **Hybrid Decision Layer**  
  Combines rule-based and ML outputs into final recommendations  

- **Risk Filtering**  
  Flags sensitive permissions for manual review  

- **Access Pattern Analysis**  
  Classifies permissions as baseline, common, rare, or unique  

## Data Pipeline

1. Ingestion – Load AD exports (CSV/XLSX)  
2. Cleaning – Parse and clean group memberships  
3. Processing – Build role-based permission matrices  
4. Recommendation – Combine deterministic + ML outputs  
