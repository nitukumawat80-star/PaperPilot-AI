
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "papers_clean.csv"

# Small, efficient model for semantic search
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    if not DATA_FILE.exists():
        print("Clean dataset not found.")
        print("First run: python src/clean_papers.py")
        return

    df = pd.read_csv(DATA_FILE).fillna("")

    if df.empty:
        print("Dataset is empty.")
        return

    # Combine title and abstract for better search
    documents = (
        df["title"].astype(str)
        + ". "
        + df["abstract"].astype(str)
    ).tolist()

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Creating paper embeddings...")
    paper_embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print(f"Embeddings created for {len(documents)} papers!")

    while True:
        query = input(
            "\nEnter your research query (or type 'exit'): "
        ).strip()

        if query.lower() == "exit":
            print("Semantic search closed.")
            break

        if not query:
            print("Please enter a query.")
            continue

        query_embedding = model.encode(
            [query],
            normalize_embeddings=True
        )

        scores = cosine_similarity(
            query_embedding,
            paper_embeddings
        )[0]
        print("Similarity scores:", scores)

        # Rank papers by similarity
        ranked_indices = scores.argsort()[::-1]
        print("Ranked paper indices:", ranked_indices)

        print("\nTop matching research papers:\n")

        for rank, idx in enumerate(ranked_indices[:5], start=1):
            row = df.iloc[idx]

            print(f"{rank}. {row['title']}")
            print(f"Similarity score: {scores[idx]:.4f}")
            print(f"Authors: {row['authors']}")
            print(f"URL: {row['url']}")
            print("-" * 70)


if __name__ == "__main__":
    main()
