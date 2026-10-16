from google import genai
import os
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "papers_clean.csv"

load_dotenv(PROJECT_ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

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


def retrieve_context(query, df, documents, model, top_k=3):
    print("Creating embeddings for research papers...")

    paper_embeddings = model.encode(
        documents,
        normalize_embeddings=True
    )

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
        row = df.iloc[idx]

        results.append({
            "title": row["title"],
            "abstract": row["abstract"],
            "url": row["url"],
            "score": float(scores[idx])
        })

    return results



def build_context(retrieved_papers):
    context_parts = []

    for index, paper in enumerate(retrieved_papers, start=1):
        context_parts.append(
            f"[SOURCE {index}]\n"
            f"Title: {paper['title']}\n"
            f"Abstract: {paper['abstract']}\n"
            f"URL: {paper['url']}"
        )

    return "\n\n".join(context_parts)






def build_prompt(query, context):
    prompt = f"""
You are PaperPilot AI, an academic research assistant.

Answer the user's question using only the research context
provided below.

STRICT RULES:
1. Support every factual claim with evidence from the context.
2. Cite sources using their exact source IDs, such as [SOURCE 1].
3. Use a source ID only when that source supports the claim.
4. Do not invent facts, statistics, quotations, or references.
5. If the context lacks evidence, clearly say so.
6. Mention limitations and uncertainty when relevant.
7. Do not treat a paper title alone as evidence for its claims.
8. Do not use outside knowledge to fill missing information.
9. Every citation must contain exactly one source ID.
10. Use only these citation formats: [SOURCE 1], [SOURCE 2],
    or [SOURCE 3], when those sources are available.
11. Never combine multiple source IDs in one bracket.
12. If multiple sources support a claim, cite them separately:
    [SOURCE 1] [SOURCE 2].
13. Do not use a source ID unless its source supports the claim.

RESEARCH CONTEXT:
{context}

USER QUESTION:
{query}

Write the response with these sections:
- Answer
- Key findings
- Limitations or missing information

Place source IDs immediately after the claims they support.
"""
    return prompt.strip()




import re





def validate_citations(answer, retrieved_papers):
    import re

    valid_ids = {
        str(number)
        for number in range(1, len(retrieved_papers) + 1)
    }

    citations = re.findall(r"\[([^\]]+)\]", answer)
    valid_count = 0
    invalid_ids = []
    invalid_formats = []

    for citation in citations:
        citation = citation.strip()

        # Combined source IDs ko detect karo
        if re.search(
            r"SOURCE\s+\d+.*[,;].*SOURCE\s+\d+",
            citation,
            re.IGNORECASE
        ):
            invalid_formats.append(citation)
            continue

        match = re.fullmatch(
            r"SOURCE\s+(\d+)",
            citation,
            re.IGNORECASE
        )

        # Normal Markdown links aur paper titles ignore karo
        if not match:
            continue

        source_id = match.group(1)

        if source_id in valid_ids:
            valid_count += 1
        else:
            invalid_ids.append(source_id)

    warnings = []

    if invalid_ids:
        warnings.append(
            "Invalid source IDs: " + ", ".join(invalid_ids)
        )

    if invalid_formats:
        warnings.append(
            "Combined citation formats found: "
            + "; ".join(invalid_formats)
        )

    if warnings:
        return answer + "\n\nCitation warning: " + " | ".join(warnings)

    if valid_count == 0:
        return answer + "\n\nCitation check: No valid source IDs detected."

    return (
        answer
        + f"\n\nCitation check: {valid_count} valid source reference(s) detected."
    )

def resolve_citations(answer, retrieved_papers):
    import re

    def replace_source(match):
        source_number = int(match.group(1))

        if 1 <= source_number <= len(retrieved_papers):
            paper = retrieved_papers[source_number - 1]
            title = paper["title"]
            url = paper["url"]

            return f"[{title}]({url})"

        return match.group(0)

    return re.sub(
        r"\[SOURCE\s+(\d+)\]",
        replace_source,
        answer,
        flags=re.IGNORECASE
    )



def generate_answer(prompt):
    if not GEMINI_API_KEY:
        return "Error: Gemini API key not found. Check your .env file."

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text or "No answer was generated."

    except Exception as e:
        print(f"Gemini API error: {e}")
        return "Sorry, I could not generate an answer."


def answer_research_question(query, df, documents, model):
    """Generate a research answer for the Streamlit frontend."""

    retrieved_papers = retrieve_context(
        query,
        df,
        documents,
        model,
        top_k=3
    )

    context = build_context(retrieved_papers)
    prompt = build_prompt(query, context)

    answer = generate_answer(prompt)

    answer = validate_citations(
        answer,
        retrieved_papers
    )

    answer = resolve_citations(
        answer,
        retrieved_papers
    )

    return answer, retrieved_papers


def main():
    if not GEMINI_API_KEY:
        print("API key not found. Check your .env file.")
        return

    df, documents = load_documents()

    if df is None:
        return

    print("Loading semantic model...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"PaperPilot AI ready for {len(documents)} papers!")

    while True:
        query = input(
            "\nAsk a research question (or type 'exit'): "
        ).strip()

        if query.lower() == "exit":
            print("PaperPilot AI closed.")
            break

        if not query:
            print("Please enter a question.")
            continue

        print("\nRetrieving relevant papers...")

        retrieved_papers = retrieve_context(
            query, df, documents, model, top_k=3
        )

        context = build_context(retrieved_papers)
        prompt = build_prompt(query, context)

        print("\nGenerating answer...")

        
       
        answer = generate_answer(prompt)

        answer = validate_citations(
            answer,
            retrieved_papers
        )

        answer = resolve_citations(
            answer,
            retrieved_papers
        )

        print("\nPaperPilot AI Answer:\n")
        print(answer)



        
    print("\nReferences:")
    for number, paper in enumerate(retrieved_papers, start=1):
        print(f"[{number}] {paper['title']}")
        print(f"    URL: {paper['url']}")
        print(f"    Retrieval similarity: {paper['score']:.4f}")
        print("-" * 70)



if __name__ == "__main__":
    main()
