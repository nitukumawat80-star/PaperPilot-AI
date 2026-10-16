import pandas as pd
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "papers_clean.csv"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_documents():
    if not DATA_FILE.exists():
        print("Clean dataset not found.")
        print("First run: python src/clean_papers.py")
        return None, None

    df = pd.read_csv(DATA_FILE).fillna("")

    if df.empty:
        print("Dataset is empty.")
        return None, None

    documents = (
        df["title"].astype(str)
        + ". "
        + df["abstract"].astype(str)
    ).tolist()

    return df, documents
def build_bm25_index(documents):
    tokenized_documents = [
        document.lower().split()
        for document in documents
    ]

    bm25 = BM25Okapi(tokenized_documents)

    return bm25

def search_bm25(query, bm25, documents, top_k=5):
    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    ranked_indices = scores.argsort()[::-1]

    results = []

    for idx in ranked_indices[:top_k]:
        results.append({
            "index": int(idx),
            "score": float(scores[idx])
        })

    return results

def build_semantic_index(documents, model):
    embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return embeddings


def search_semantic(query, model, paper_embeddings, top_k=5):
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    scores = cosine_similarity(
        query_embedding,
        paper_embeddings
    )[0]

    ranked_indices = scores.argsort()[::-1]

    results = []

    for idx in ranked_indices[:top_k]:
        results.append({
            "index": int(idx),
            "score": float(scores[idx])
        })

    return results

def combine_results(bm25_results, semantic_results, k=60):
    fused_scores = {}

    for rank, result in enumerate(bm25_results, start=1):
        idx = result["index"]
        fused_scores[idx] = fused_scores.get(idx, 0) + 1 / (k + rank)

    for rank, result in enumerate(semantic_results, start=1):
        idx = result["index"]
        fused_scores[idx] = fused_scores.get(idx, 0) + 1 / (k + rank)

    ranked_results = sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return [
        {"index": idx, "score": score}
        for idx, score in ranked_results
    ]

def main():
    df, documents = load_documents()

    if df is None:
        return

    print("Building BM25 index...")
    bm25 = build_bm25_index(documents)

    print("Loading semantic model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Creating paper embeddings...")
    paper_embeddings = build_semantic_index(documents, model)

    print(f"Hybrid search ready for {len(documents)} papers!")

    while True:
        query = input(
            "\nEnter your research query (or type 'exit'): "
        ).strip()

        if query.lower() == "exit":
            print("Hybrid search closed.")
            break

        if not query:
            print("Please enter a query.")
            continue

        bm25_results = search_bm25(query, bm25, documents)
        semantic_results = search_semantic(
            query, model, paper_embeddings
        )

        hybrid_results = combine_results(
            bm25_results, semantic_results
        )

        print("\nTop matching research papers:\n")

        for rank, result in enumerate(hybrid_results[:5], start=1):
            idx = result["index"]
            row = df.iloc[idx]

            print(f"{rank}. {row['title']}")
            print(f"RRF score: {result['score']:.4f}")
            print(f"Authors: {row['authors']}")
            print(f"URL: {row['url']}")
            print("-" * 70)


if __name__ == "__main__":
    main()
