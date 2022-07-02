"""Query Wikidata for specific page-type Wikidata entity URIs"""
import requests, sys, urllib, tqdm
import pandas as pd

url = "https://query.wikidata.org/sparql"
query_templates = {
    "disambig": """SELECT DISTINCT ?s WHERE {
        ?s wdt:P31 wd:Q4167410 .
        ?page schema:about ?s .
        ?page schema:inLanguage "%s" .
    }""",
    "list": """SELECT DISTINCT ?s WHERE {
        {?s wdt:P31 wd:Q13406463 .}  UNION  {?s wdt:P360 ?l . }
        ?page schema:about ?s .
        ?page schema:inLanguage "%s" .
    }""",
}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("type", choices=["disambig", "list"])
    parser.add_argument("language", type=str, help="Language code")
    args = parser.parse_args()

    query = query_templates[args.type] % args.language

    response = requests.get(
        url,
        params={"query": query},
        headers={
            "Accept": "text/csv",
            "user-agent": "vu-amsterdam-entity-linking/0.0.1",
        },
    )
    if response.ok:
        lines = response.text.splitlines()
        if lines:
            for line in lines[1:]:
                print(line.replace("http://www.wikidata.org/entity/", ""))
