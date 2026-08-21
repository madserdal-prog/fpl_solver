"""
Look up player IDs from names in master_data.json -- for use when
fetch_my_team.py doesn't work yet (before the GW1 deadline has passed).

Usage: python find_player_ids.py Pickford O'Reilly Senesi Lacroix ...
"""
import json
import sys

with open("master_data.json", encoding="utf-8") as f:
    master = json.load(f)

for name_query in sys.argv[1:]:
    matches = [e for e in master["elements"]
               if name_query.lower() in e["web_name"].lower()]
    if not matches:
        print(f"'{name_query}' -> NO MATCH (check spelling)")
    elif len(matches) == 1:
        e = matches[0]
        print(f"'{name_query}' -> id={e['id']}  ({e['web_name']}, {e['team']}, {e['element_type']})")
    else:
        print(f"'{name_query}' -> MULTIPLE MATCHES, pick the right one:")
        for e in matches:
            print(f"    id={e['id']}  ({e['web_name']}, {e['team']}, {e['element_type']})")
