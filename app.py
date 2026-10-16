
import sys
from pathlib import Path

import streamlit as st
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# 1. Project paths and imports
# --------------------------------------------------

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


# --------------------------------------------------
# 2. Streamlit page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="PaperPilot AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp {
        background-color: #0b1120;
        color: #e5e7eb;
    }

    .main-title {
        font-size: 40px;
        font-weight: 800;
        color: #8bafff;
        margin-bottom: 0;
    }

    .subtitle {
        color: #a8b3c7;
        font-size: 17px;
    }

    .feature-card {
        background: #151f33;
        padding: 20px;
        border: 1px solid #293750;
        border-radius: 14px;
        min-height: 145px;
    }

    .feature-card h3 {
        color: #a5c1ff;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. Load data and models only once
# --------------------------------------------------

@st.cache_resource
def initialize_research():
    df, documents = load_documents()

    if df is None:
        return None, None, None

    model = SentenceTransformer(MODEL_NAME)

    return df, documents, model


@st.cache_resource
def initialize_hybrid_search():
    df, documents = load_documents()

    if df is None:
        return None, None, None, None, None

    bm25 = build_bm25_index(documents)

    model = SentenceTransformer(MODEL_NAME)

    embeddings = build_semantic_index(documents, model)

    return df, documents, bm25, model, embeddings


# --------------------------------------------------
# 4. Sidebar navigation
# --------------------------------------------------

with st.sidebar:
    st.title("📚 PaperPilot AI")
    st.caption("Your AI Research Companion")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🔎 Paper Search",
            "🤖 AI Research Assistant",
            "ℹ️ About",
        ],
    )

    st.divider()
    st.caption("Semantic Search + BM25 + Gemini RAG")


# --------------------------------------------------
# 5. Dashboard
# --------------------------------------------------

if page == "🏠 Dashboard":

    st.markdown(
        '<p class="main-title">PaperPilot AI</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="subtitle">Discover research. Understand faster. '
        'Explore knowledge with AI.</p>',
        unsafe_allow_html=True,
    )

    st.write("")

    dashboard_query = st.text_input(
        "🔎 What do you want to research?",
        placeholder="e.g. Retrieval-Augmented Generation",
        key="dashboard_query",
    )

    if st.button("Start Research", type="primary"):
        if dashboard_query.strip():
            st.session_state["assistant_query"] = dashboard_query.strip()
            st.session_state["navigation_target"] = (
                "🤖 AI Research Assistant"
            )
            st.success(
                "Question saved! Open AI Research Assistant "
                "from the sidebar to generate your answer."
            )
        else:
            st.warning("Please enter a research question.")

    st.write("")
    st.subheader("Explore PaperPilot AI")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🔍 Smart Search</h3>
            Find relevant papers using keywords and semantic meaning.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🤖 AI Answers</h3>
            Generate research answers with Gemini and retrieved context.
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>📑 References</h3>
            Explore paper abstracts, relevance scores and original links.
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.info(
        "PaperPilot AI combines research retrieval with AI-powered "
        "question answering."
    )


# --------------------------------------------------
# 6. Hybrid Paper Search
# --------------------------------------------------

elif page == "🔎 Paper Search":

    st.title("🔎 Smart Paper Search")

    st.write(
        "Search research papers using keyword matching and semantic similarity."
    )

    search_query = st.text_input(
        "Enter research keywords or a question",
        placeholder="e.g. machine learning for healthcare",
        key="hybrid_search_query",
    )

    search_limit = st.selectbox(
        "Number of results",
        [3, 5, 10],
        index=1,
    )

    if st.button("Search Papers", type="primary"):

        if not search_query.strip():
            st.warning("Please enter a search query.")

        else:
            try:
                with st.spinner("Preparing hybrid search..."):
                    (
                        df,
                        documents,
                        bm25,
                        model,
                        embeddings,
                    ) = initialize_hybrid_search()

                if df is None:
                    st.error(
                        "Dataset unavailable. Run "
                        "`python src/clean_papers.py` first."
                    )

                else:
                    with st.spinner("Finding relevant research papers..."):

                        bm25_results = search_bm25(
                            search_query.strip(),
                            bm25,
                            documents,
                            top_k=search_limit,
                        )

                        semantic_results = search_semantic(
                            search_query.strip(),
                            model,
                            embeddings,
                            top_k=search_limit,
                        )

                        hybrid_results = combine_results(
                            bm25_results,
                            semantic_results,
                        )

                    st.session_state["hybrid_results"] = [
                        {
                            **result,
                            "title": str(
                                df.iloc[result["index"]]["title"]
                            ),
                            "abstract": str(
                                df.iloc[result["index"]]["abstract"]
                            ),
                            "authors": str(
                                df.iloc[result["index"]]["authors"]
                            ),
                            "url": str(
                                df.iloc[result["index"]]["url"]
                            ),
                        }
                        for result in hybrid_results[:search_limit]
                    ]

                    st.session_state["hybrid_query"] = search_query.strip()

            except Exception as exc:
                st.error(f"Search failed: {exc}")

    if "hybrid_results" in st.session_state:
        st.subheader(
            f"Results for: {st.session_state.get('hybrid_query', '')}"
        )

        results = st.session_state["hybrid_results"]

        if not results:
            st.info("No matching papers found.")

        for rank, paper in enumerate(results, start=1):
            with st.container(border=True):
                st.markdown(f"### {rank}. {paper['title']}")

                st.caption(f"Authors: {paper['authors']}")

                st.write(paper["abstract"])

                st.metric(
                    "Hybrid RRF Score",
                    f"{paper['score']:.4f}",
                )

                if paper["url"]:
                    st.markdown(
                        f"[Open original paper]({paper['url']})"
                    )


# --------------------------------------------------
# 7. AI Research Assistant
# --------------------------------------------------

elif page == "🤖 AI Research Assistant":

    st.title("🤖 AI Research Assistant")

    st.write(
        "Ask a question and receive an answer based on retrieved papers."
    )

    default_query = st.session_state.get("assistant_query", "")

    question = st.text_area(
        "Your research question",
        value=default_query,
        height=110,
        key="rag_question",
        placeholder="What are the main benefits of RAG?",
    )

    answer_top_k = st.selectbox(
        "Number of papers for answer context",
        [3],
        index=0,
    )

    if st.button("Generate Research Answer", type="primary"):

        if not question.strip():
            st.warning("Please enter a research question.")

        else:
            try:
                with st.spinner("Loading research model..."):
                    df, documents, model = initialize_research()

                if df is None:
                    st.error(
                        "Dataset unavailable. Run "
                        "`python src/clean_papers.py` first."
                    )

                else:
                    with st.spinner(
                        "Retrieving papers and generating AI answer..."
                    ):
                        answer, papers = answer_research_question(
                            question.strip(),
                            df,
                            documents,
                            model,
                        )

                    st.session_state["research_answer"] = answer
                    st.session_state["research_papers"] = papers

            except Exception as exc:
                st.error(f"Research generation failed: {exc}")

    if "research_answer" in st.session_state:

        st.divider()
        st.subheader("🧠 AI Research Answer")

        st.markdown(st.session_state["research_answer"])

        st.subheader("📑 Retrieved Research Papers")

        for number, paper in enumerate(
            st.session_state["research_papers"],
            start=1,
        ):
            with st.expander(f"{number}. {paper['title']}"):

                st.write("**Abstract**")
                st.write(paper["abstract"])

                st.write(
                    f"**Retrieval similarity:** {paper['score']:.4f}"
                )

                if paper["url"]:
                    st.markdown(
                        f"[Open original paper]({paper['url']})"
                    )


# --------------------------------------------------
# 8. About
# --------------------------------------------------

else:

    st.title("ℹ️ About PaperPilot AI")

    st.write("""
    PaperPilot AI is an AI-powered research intelligence platform
    designed to help users discover and understand research papers.

    **Current capabilities**
    - BM25 keyword-based paper search
    - Semantic search using sentence embeddings
    - Hybrid ranking using Reciprocal Rank Fusion
    - Research question answering with Gemini
    - Retrieved paper abstracts and original links

    **Technology stack**
    - Python
    - Streamlit
    - Sentence Transformers
    - BM25
    - Google Gemini API
    - Pandas
    """)
