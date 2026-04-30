from DataLayer.loader import DataLoader
from DataLayer.cleaner import DataCleaner
from DeterministicLayer.rules_recommender import RulesRecommender


loader = DataLoader(base_path="data/raw")
cleaner = DataCleaner()

df = loader.load_file("ce_ad_user_rights_all.xlsx")
df = cleaner.clean_groups(df)

recommender = RulesRecommender(min_confidence=0.6)

results = recommender.recommend_for_new_user(
    users_df=df,
    title="Your Job Title Here",
    department="Your Department Here"
)

print(results.head(30))