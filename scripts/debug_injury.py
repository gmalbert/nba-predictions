import requests, json, sys

url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
resp = requests.get(url, timeout=15)
print("HTTP status:", resp.status_code)
data = resp.json()

top_keys = list(data.keys())
print("Top-level keys:", top_keys)

entries = data.get("injuries", [])
print("Number of team entries:", len(entries))

if entries:
    first = entries[0]
    print("First entry keys:", list(first.keys()))
    print("First entry 'team' value:", json.dumps(first.get("team"), indent=2))
    injs = first.get("injuries", [])
    print("Injuries count in first entry:", len(injs))
    if injs:
        print("First injury keys:", list(injs[0].keys()))
        print("First injury sample:", json.dumps({k: injs[0][k] for k in list(injs[0].keys())[:6]}, indent=2))
