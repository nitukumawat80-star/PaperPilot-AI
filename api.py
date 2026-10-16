
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_pipeline import (
    load_documents,
    answer_research_question,
    MODEL_NAME,
)

from hybrid_search import (
    build_bm25_index,
    search_bm25,
    build_semantic_index,
    search_semantic,
    combine_results,
)

app = FastAPI(
    title="PaperPilot AI API",
    description="Research paper search and AI research assistant",
    version="1.0.0",
)

# Local development only. Production mein apne deployed
# frontend ka exact origin configure karna.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)


class QuestionRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)


@app.on_event("startup")
def initialize_backend():
    df, documents = load_documents()

    if df is None:
        raise RuntimeError(
            "Research dataset missing. Run the data collection "
            "and cleaning scripts first."
        )

    model = SentenceTransformer(MODEL_NAME)
    bm25 = build_bm25_index(documents)
    embeddings = build_semantic_index(documents, model)

    app.state.df = df
    app.state.documents = documents
    app.state.model = model
    app.state.bm25 = bm25
    app.state.embeddings = embeddings


@app.get("/")
def root():
    return {
        "message": "PaperPilot AI API is running",
        "docs": "/docs",
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "papers_loaded": len(app.state.documents),
    }


@app.post("/api/search")
def search_papers(request: SearchRequest):
    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query is empty")

    df = app.state.df

    keyword_results = search_bm25(
        query,
        app.state.bm25,
        app.state.documents,
        top_k=request.top_k,
    )

    semantic_results = search_semantic(
        query,
        app.state.model,
        app.state.embeddings,
        top_k=request.top_k,
    )

    hybrid_results = combine_results(
        keyword_results,
        semantic_results,
    )[:request.top_k]

    papers = []

    for result in hybrid_results:
        row = df.iloc[result["index"]]

        papers.append({
            "title": str(row.get("title", "")),
            "authors": str(row.get("authors", "")),
            "abstract": str(row.get("abstract", "")),
            "url": str(row.get("url", "")),
            "score": round(float(result["score"]), 6),
        })

    return {
        "query": query,
        "count": len(papers),
        "results": papers,
    }


@app.post("/api/ask")
def ask_question(request: QuestionRequest):
    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Question is empty")

    answer, papers = answer_research_question(
        query,
        app.state.df,
        app.state.documents,
        app.state.model,
    )

    return {
        "question": query,
        "answer": answer,
        "sources": [
            {
                "title": paper["title"],
                "abstract": paper["abstract"],
                "url": paper["url"],
                "score": round(float(paper["score"]), 6),
            }
            for paper in papers
        ],
    }
