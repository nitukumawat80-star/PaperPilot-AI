
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "papers.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "papers_clean.csv"


def clean_papers():
    print("Loading research papers...")

    df = pd.read_csv(INPUT_FILE)

    initial_count = len(df)

    # Remove leading and trailing spaces
    for column in ["paper_id", "title", "authors", "abstract", "url"]:
        df[column] = df[column].fillna("").astype(str).str.strip()

    # Remove duplicate papers
    df = df.drop_duplicates(subset=["paper_id"])

    # Remove records without essential information
    df = df[
        (df["paper_id"] != "")
        & (df["title"] != "")
        & (df["abstract"] != "")
        & (df["url"] != "")
    ].copy()

    # Convert publication dates into a consistent format
    df["published"] = pd.to_datetime(
        df["published"], errors="coerce", utc=True
    )

    # Remove records with invalid publication dates
    df = df.dropna(subset=["published"])

    # Save clean data
    df["published"] = df["published"].dt.strftime("%Y-%m-%d")

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\nData cleaning completed!")
    print(f"Original papers: {initial_count}")
    print(f"Clean papers: {len(df)}")
    print(f"Removed records: {initial_count - len(df)}")
    print(f"Saved file: {OUTPUT_FILE}")

    print("\nClean paper titles:")
    for index, title in enumerate(df["title"], start=1):
        print(f"{index}. {title}")


if __name__ == "__main__":
    clean_papers()
