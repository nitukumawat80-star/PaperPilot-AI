
import time
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path

API_URL = "https://export.arxiv.org/api/query"

HEADERS = {
    "User-Agent": "PaperPilotAI/1.0 (student research project)"
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "papers.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)


def collect_papers(max_results=100):
    params = {
        "search_query": (
            'all:"retrieval augmented generation" '
            "OR all: RAG "
            "OR all:LLM "
            "OR all:transformer"
        ),
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    print("Fetching research papers from arXiv...")
    print(f"Maximum papers requested: {max_results}")

    time.sleep(3)

    response = requests.get(
        API_URL,
        params=params,
        headers=HEADERS,
        timeout=60,
    )
    response.raise_for_status()

    root = ET.fromstring(response.content)

    namespace = {
        "atom": "http://www.w3.org/2005/Atom"
    }

    papers = []

    for entry in root.findall("atom:entry", namespace):
        title = " ".join(
            entry.findtext(
                "atom:title",
                default="",
                namespaces=namespace,
            ).split()
        )

        abstract = " ".join(
            entry.findtext(
                "atom:summary",
                default="",
                namespaces=namespace,
            ).split()
        )

        paper_id = entry.findtext(
            "atom:id",
            default="",
            namespaces=namespace,
        ).strip()

        published = entry.findtext(
            "atom:published",
            default="",
            namespaces=namespace,
        )

        authors = [
            author.findtext(
                "atom:name",
                default="",
                namespaces=namespace,
            )
            for author in entry.findall("atom:author", namespace)
        ]

        if not paper_id or not title:
            continue

        papers.append({
            "paper_id": paper_id,
            "title": title,
            "authors": ", ".join(authors),
            "abstract": abstract,
            "published": published,
            "url": paper_id,
        })

    if not papers:
        print("No papers found. Try again later.")
        return

    new_df = pd.DataFrame(papers)

    if OUTPUT_FILE.exists():
        old_df = pd.read_csv(OUTPUT_FILE).fillna("")
        combined_df = pd.concat(
            [old_df, new_df],
            ignore_index=True,
        )
    else:
        combined_df = new_df

    combined_df.drop_duplicates(
        subset=["paper_id"],
        keep="last",
        inplace=True,
    )

    combined_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nNew papers fetched: {len(new_df)}")
    print(f"Total unique papers saved: {len(combined_df)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("\nSample research paper titles:\n")

    for index, title in enumerate(
        combined_df["title"].head(10),
        start=1,
    ):
        print(f"{index}. {title}")


if __name__ == "__main__":
    try:
        collect_papers(max_results=100)
    except requests.RequestException as error:
        print(f"API request failed: {error}")
    except ET.ParseError as error:
        print(f"Could not parse arXiv response: {error}")
