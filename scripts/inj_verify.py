import sys
sys.path.insert(0, ".")
from utils.data_fetcher import get_injury_report

df = get_injury_report()
print("rows:", len(df))
print(df[["team", "player_name", "status"]].head(10).to_string())
empty = (df["team"] == "").sum()
print("empty team rows:", empty)
