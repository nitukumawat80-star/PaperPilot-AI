import sys
from pathlib import Path

import streamlit as st
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_pipeline import load_documents, answer_research_question, MODEL_NAME
from hybrid_search import (
    build_bm25_index,
    search_bm25,
    build_semantic_index,
    search_semantic,
    combine_results,
)

st.set_page_config(
    page_title="PaperPilot AI | Research Workspace",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root { --muted:#9ba9c7; --line:rgba(164,183,231,.15); }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:radial-gradient(ellipse at 12% 0%,rgba(91,73,190,.22),transparent 34%),radial-gradient(ellipse at 90% 5%,rgba(37,157,196,.13),transparent 30%),#080d19; color:#eaf0ff; }
[data-testid="stHeader"] { background:rgba(8,13,25,.75); }
[data-testid="stSidebar"] { background:#0b1120; border-right:1px solid var(--line); }
[data-testid="stSidebar"] > div { padding-top:1.3rem; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.035em; color:#f4f6ff; }
p,label,.stMarkdown { color:#dce5fa; }
.brand { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.45rem; letter-spacing:-.05em; color:#f6f7ff; }
.brand span { color:#9b8cff; }
.eyebrow { color:#9f91ff; font-size:.76rem; font-weight:700; text-transform:uppercase; letter-spacing:.14em; margin-bottom:.55rem; }
.hero { background:linear-gradient(120deg,rgba(28,37,68,.96),rgba(18,26,48,.88)); border:1px solid rgba(153,142,255,.25); border-radius:24px; padding:2.1rem 2.2rem; margin:.35rem 0 1.4rem; box-shadow:0 18px 55px rgba(0,0,0,.18); }
.hero h1 { font-size:clamp(2rem,4vw,3.25rem); line-height:1.06; margin:.1rem 0 .8rem; }
.hero p,.subtitle { color:#b8c5e3; max-width:720px; font-size:1.02rem; }
.gradient-text { background:linear-gradient(90deg,#b8aaff,#65e3f6); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.metric { background:linear-gradient(145deg,rgba(21,31,57,.96),rgba(13,21,39,.96)); border:1px solid var(--line); padding:1rem 1.1rem; border-radius:17px; min-height:105px; }
.metric-label { color:#9ba9c7; font-size:.82rem; margin-bottom:.45rem; }
.metric-value { color:#f4f6ff; font-family:'Space Grotesk',sans-serif; font-size:1.55rem; font-weight:700; }
.metric-note,.side-note { color:#8191b3; font-size:.78rem; margin-top:.25rem; }
.feature-card { background:linear-gradient(145deg,rgba(19,29,52,.98),rgba(13,21,39,.98)); border:1px solid var(--line); border-radius:17px; padding:1.15rem; min-height:140px; }
.feature-card h3 { color:#b8aaff; }
.paper-card { background:linear-gradient(145deg,rgba(19,29,52,.98),rgba(13,21,39,.98)); border:1px solid var(--line); border-radius:17px; padding:1.15rem 1.25rem; margin:.55rem 0; }
.paper-title { font-family:'Space Grotesk',sans-serif; font-weight:600; color:#f2f5ff; font-size:1.05rem; line-height:1.4; margin-bottom:.55rem; }
.pill { display:inline-block; padding:.25rem .55rem; border-radius:999px; border:1px solid rgba(139,124,255,.35); background:rgba(139,124,255,.11); color:#c4bbff; font-size:.73rem; margin:0 .3rem .3rem 0; }
div.stButton > button, div.stFormSubmitButton > button { border-radius:11px; border:1px solid rgba(155,140,255,.45); background:linear-gradient(100deg,#7565ed,#4c9ecb); color:white; font-weight:700; min-height:2.7rem; }
div.stButton > button:hover, div.stFormSubmitButton > button:hover { border-color:#b5aaff; color:white; filter:brightness(1.08); }
.stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] { background:#0d1528 !important; color:#edf2ff !important; border-color:rgba(164,183,231,.2) !important; border-radius:11px !important; }
[data-testid="stExpander"] { border:1px solid var(--line); border-radius:12px; background:rgba(17,26,48,.55); }
hr { border-color:var(--line); }
</style>
""", unsafe_allow_html=True)


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


if "saved_papers" not in st.session_state:
    st.session_state.saved_papers = []
if "navigation_target" not in st.session_state:
    st.session_state.navigation_target = "Overview"
if "assistant_query" not in st.session_state:
    st.session_state.assistant_query = ""

with st.sidebar:
    st.markdown('<div class="brand">PaperPilot<span> AI</span> ✦</div>', unsafe_allow_html=True)
    st.caption("RESEARCH INTELLIGENCE WORKSPACE")
    st.divider()
    pages = ["Overview", "Explore Papers", "AI Research Copilot", "My Library", "About"]
    default_index = pages.index(st.session_state.navigation_target) if st.session_state.navigation_target in pages else 0
    page = st.radio("WORKSPACE", pages, index=default_index, key="workspace_navigation")
    st.session_state.navigation_target = page
    st.divider()
    st.markdown("**Your research toolkit**")
    st.markdown('<div class="side-note">Semantic discovery · Hybrid retrieval · Gemini-powered RAG</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-note">Built for students, researchers and curious minds.</div>', unsafe_allow_html=True)


def render_paper(paper, rank, show_save=True):
    title = str(paper.get("title", "Untitled research paper"))
    authors = str(paper.get("authors", "Authors not listed"))
    abstract = str(paper.get("abstract", ""))
    url = str(paper.get("url", ""))
    score = paper.get("score", None)
    with st.container(border=True):
        st.markdown(f'<div class="paper-title">{rank}. {title}</div>', unsafe_allow_html=True)
        st.markdown(f'<span class="pill">{authors[:110]}</span>', unsafe_allow_html=True)
        if score is not None:
            try:
                st.markdown(f'<span class="pill">Relevance · {float(score):.4f}</span>', unsafe_allow_html=True)
            except (ValueError, TypeError):
                pass
        if abstract and abstract.lower() != "nan":
            st.write(abstract)
        if url and url.lower() != "nan":
            st.markdown(f"[Open original paper ↗]({url})")
        if show_save:
            if title in st.session_state.saved_papers:
                st.caption("✓ Saved to My Library")
            elif st.button("＋ Save paper", key=f"save_{rank}_{title[:40]}"):
                st.session_state.saved_papers.append(title)
                st.rerun()


if page == "Overview":
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">YOUR AI RESEARCH PARTNER</div>
      <h1>From research overload<br>to <span class="gradient-text">clear insights.</span></h1>
      <p>Discover relevant papers, ask research questions, and explore academic knowledge in one focused workspace — with answers grounded in paper sources.</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        df, documents, model = initialize_research()
        paper_count = len(df) if df is not None else 0
    except Exception:
        paper_count = "—"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="metric"><div class="metric-label">PAPERS IN WORKSPACE</div><div class="metric-value">{paper_count}</div><div class="metric-note">Searchable research collection</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric"><div class="metric-label">DISCOVERY ENGINE</div><div class="metric-value">Hybrid AI</div><div class="metric-note">Keyword + semantic retrieval</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric"><div class="metric-label">RESEARCH COPILOT</div><div class="metric-value">RAG</div><div class="metric-note">Answers grounded in retrieved papers</div></div>', unsafe_allow_html=True)

    st.markdown('<h3 style="margin-top:1.5rem">What are you researching today?</h3><p class="subtitle">Start with a topic or ask the AI copilot a research question.</p>', unsafe_allow_html=True)
    with st.form("overview_search"):
        query = st.text_input("Research topic", placeholder="e.g. retrieval augmented generation, explainable AI, medical imaging…", label_visibility="collapsed")
        submitted = st.form_submit_button("Explore research ↗", use_container_width=True)
    if submitted:
        if query.strip():
            st.session_state.assistant_query = query.strip()
            st.session_state.navigation_target = "AI Research Copilot"
            st.rerun()
        else:
            st.warning("Please enter a research topic first.")

    st.markdown("**Try a research question**")
    prompts = [
        "What is retrieval-augmented generation?",
        "How are transformers used in medical imaging?",
        "What are current challenges in explainable AI?",
    ]
    cols = st.columns(3)
    for i, prompt in enumerate(prompts):
        with cols[i]:
            if st.button(prompt, key=f"prompt_{i}", use_container_width=True):
                st.session_state.assistant_query = prompt
                st.session_state.navigation_target = "AI Research Copilot"
                st.rerun()

    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown('<div class="feature-card"><h3>⌕ Smart Search</h3><p>Find relevant papers using keywords and semantic meaning.</p></div>', unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="feature-card"><h3>✦ AI Answers</h3><p>Generate research answers with Gemini and retrieved context.</p></div>', unsafe_allow_html=True)
    with col_c:
        st.markdown('<div class="feature-card"><h3>↗ Source Papers</h3><p>Inspect abstracts, relevance scores and original paper links.</p></div>', unsafe_allow_html=True)

elif page == "Explore Papers":
    st.markdown('<div class="eyebrow">DISCOVER KNOWLEDGE</div><h1>Explore Papers</h1><p class="subtitle">Search your collection with keywords or research concepts.</p>', unsafe_allow_html=True)
    with st.form("paper_search"):
        search_query = st.text_input("Search query", placeholder="e.g. machine learning for healthcare")
        search_limit = st.selectbox("Number of results", [3, 5, 10], index=1)
        search_submit = st.form_submit_button("Search papers ↗", use_container_width=True)

    if search_submit:
        if not search_query.strip():
            st.warning("Please enter a search query.")
        else:
            try:
                with st.spinner("Preparing hybrid search…"):
                    df, documents, bm25, model, embeddings = initialize_hybrid_search()
                if df is None:
                    st.error("Dataset unavailable. Run `python src/clean_papers.py` first.")
                else:
                    with st.spinner("Finding relevant research papers…"):
                        bm25_results = search_bm25(search_query.strip(), bm25, documents, top_k=search_limit)
                        semantic_results = search_semantic(search_query.strip(), model, embeddings, top_k=search_limit)
                        hybrid_results = combine_results(bm25_results, semantic_results)
                    st.session_state.hybrid_results = [
                        {
                            **result,
                            "title": str(df.iloc[result["index"]]["title"]),
                            "abstract": str(df.iloc[result["index"]]["abstract"]),
                            "authors": str(df.iloc[result["index"]]["authors"]),
                            "url": str(df.iloc[result["index"]]["url"]),
                        }
                        for result in hybrid_results[:search_limit]
                    ]
                    st.session_state.hybrid_query = search_query.strip()
            except Exception as exc:
                st.error(f"Search failed: {exc}")

    if "hybrid_results" in st.session_state:
        st.markdown(f'<h3>Results for “{st.session_state.get("hybrid_query", "")}”</h3>', unsafe_allow_html=True)
        results = st.session_state.hybrid_results
        if not results:
            st.info("No matching papers found. Try a broader query.")
        for rank, paper in enumerate(results, start=1):
            render_paper(paper, rank)

elif page == "AI Research Copilot":
    st.markdown('<div class="eyebrow">ASK · SYNTHESIZE · CITE</div><h1>AI Research Copilot</h1><p class="subtitle">Ask a question and get a research-focused answer based on papers in your collection.</p>', unsafe_allow_html=True)
    with st.container(border=True):
        question = st.text_area(
            "Your research question",
            value=st.session_state.assistant_query,
            height=125,
            placeholder="Compare the main approaches to RAG and explain their limitations…",
            key="copilot_question_input",
        )
        generate = st.button("Generate research answer ✦", type="primary", use_container_width=True)

    if generate:
        if not question.strip():
            st.warning("Please enter a research question.")
        else:
            st.session_state.assistant_query = question.strip()
            try:
                with st.spinner("Retrieving papers and generating an answer…"):
                    df, documents, model = initialize_research()
                    if df is None:
                        st.error("Dataset unavailable. Run `python src/clean_papers.py` first.")
                    else:
                        answer, papers = answer_research_question(question.strip(), df, documents, model)
                        st.session_state.research_answer = answer
                        st.session_state.research_papers = papers
            except Exception as exc:
                st.error(f"Research generation failed: {exc}")

    if "research_answer" in st.session_state:
        st.markdown('<h3>✦ Research synthesis</h3>', unsafe_allow_html=True)
        st.markdown(st.session_state.research_answer)
        st.markdown('<h3>↗ Retrieved source papers</h3>', unsafe_allow_html=True)
        st.caption("Review the original papers before using this synthesis in academic work.")
        for rank, paper in enumerate(st.session_state.get("research_papers", []), start=1):
            with st.expander(f"{rank}. {paper.get('title', 'Research paper')}"):
                st.write("**Abstract**")
                st.write(paper.get("abstract", "Abstract unavailable"))
                if paper.get("score") is not None:
                    st.write(f"**Retrieval similarity:** {paper['score']:.4f}")
                if paper.get("url"):
                    st.markdown(f"[Open original paper ↗]({paper['url']})")

elif page == "My Library":
    st.markdown('<div class="eyebrow">YOUR READING LIST</div><h1>My Library</h1><p class="subtitle">Save interesting papers from Explore Papers to build a shortlist.</p>', unsafe_allow_html=True)
    if not st.session_state.saved_papers:
        st.info("Your library is empty. Open Explore Papers and use “Save paper” to add papers.")
    else:
        st.caption(f"{len(st.session_state.saved_papers)} saved paper(s)")
        for i, title in enumerate(list(st.session_state.saved_papers)):
            left, right = st.columns([5, 1])
            with left:
                st.write(f"📄 {title}")
            with right:
                if st.button("Remove", key=f"remove_{i}"):
                    st.session_state.saved_papers.remove(title)
                    st.rerun()

else:
    st.markdown('<div class="eyebrow">ABOUT THE PROJECT</div><h1>Research, made navigable.</h1>', unsafe_allow_html=True)
    st.write("PaperPilot AI is an AI-powered research intelligence platform designed to help users discover and understand academic papers.")
    st.markdown("""
    **Core capabilities**
    - BM25 keyword-based paper search
    - Semantic search using sentence embeddings
    - Hybrid ranking using Reciprocal Rank Fusion
    - Research question answering with Gemini
    - Retrieved paper abstracts and original links

    **Technology stack:** Python · Streamlit · Sentence Transformers · BM25 · Google Gemini API · Pandas
    """)
    st.info("AI-generated answers can miss context. Verify important claims against the original papers.")

st.markdown("---")
st.markdown('<div class="side-note">PaperPilot AI · Research workspace · Always verify important claims against the original paper.</div>', unsafe_allow_html=True)
