import pandas as pd

from DataLayer.cleaner import DataCleaner
from ProductLayer.hybrid_recommender import HybridRecommender

cleaner = DataCleaner()
users_df = cleaner.load_cleaned()

recommender = HybridRecommender(min_rules_confidence=0.6)

results = recommender.recommend(
    users_df=users_df,
    sam_account_name="NEW_USER_OR_EXISTING_TEST_USER",
    title="Academic Outreach & Sales Rep",
    department="CE Academic Outreach & Sales",
    top_n_users=5,
    min_ml_support=2,
    include_supervisors=False,
)

print(results[[
    "GroupName",
    "FinalDecision",
    "HybridScore",
    "RulesConfidence",
    "MLConfidence",
    "RiskLevel",
    "Reason",
]].head(50))