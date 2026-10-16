# PaperPilot AI 📚

**An AI-powered research assistant for discovering scientific papers and generating context-grounded answers.**

PaperPilot AI helps users explore research literature using keyword-based retrieval, semantic similarity, and Retrieval-Augmented Generation (RAG). It combines a research-paper dataset with AI-generated answers and source references to make academic exploration more convenient.

## ✨ Features

- **Hybrid Research Search:** Combines BM25 keyword retrieval with semantic search.
- **Semantic Search:** Uses Sentence Transformers to find papers based on the meaning of a query.
- **AI Research Assistant:** Uses Google's Gemini model to generate answers from retrieved research context.
- **Source-Aware Responses:** Returns relevant paper sources to help users investigate the supporting literature.
- **Research Paper Dataset:** Loads cleaned paper data from CSV files.
- **Streamlit Interface:** Provides a dashboard, paper-search page, AI assistant, and project information.
- **FastAPI Backend:** Exposes endpoints for health checks, research search, and AI-powered questions.

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| AI-generated answers | Google Gemini API |
| Embeddings | Sentence Transformers |
| Semantic similarity | Scikit-learn |
| Keyword retrieval | BM25 via `rank-bm25` |
| Backend API | FastAPI |
| Interactive application | Streamlit |
| Data processing | Pandas |
| Dataset format | CSV |

## 🧠 How It Works

1. **Load data:** Read research papers from `data/papers_clean.csv`.
2. **Prepare representations:** Combine paper titles and abstracts and generate semantic embeddings.
3. **Retrieve papers:** Use BM25 for keyword matching and semantic similarity for meaning-based retrieval.
4. **Combine rankings:** Apply Reciprocal Rank Fusion (RRF) to combine the two retrieval rankings.
5. **Generate an answer:** Retrieve relevant research context and pass it to Gemini.
6. **Present results:** Display the answer and associated paper sources.

## 🏗️ Project Structure

```text
PaperPilot-AI/
├── api.py
├── app.py
├── requirements.txt
├── .env                  # Create locally; never commit secrets
├── data/
│   ├── papers.csv
│   └── papers_clean.csv
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
└── src/
    ├── collect_papers.py
    ├── clean_papers.py
    ├── semantic_search.py
    ├── hybrid_search.py
    └── rag_pipeline.py
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/nitukumawat80-star/PaperPilot-AI.git
cd PaperPilot-AI
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Replace the placeholder with your own API key. Never publish the key or commit the `.env` file to GitHub.

### 5. Run the Streamlit application

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser.

### 6. Run the FastAPI backend

Open a separate terminal in the project root and activate the virtual environment. Then run:

```bash
python -m uvicorn api:app --reload --port 8000
```

API base URL: `http://127.0.0.1:8000`

Interactive API documentation: `http://127.0.0.1:8000/docs`

### 7. Run the standalone frontend

With the FastAPI backend running, open another terminal and execute:

```bash
python -m http.server 5500 --directory frontend
```

Open `http://127.0.0.1:5500/index.html`.

Keep the frontend and backend servers running in separate terminals while testing.

## 🔌 API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API status and documentation path |
| GET | `/api/health` | Health check and loaded-paper count |
| POST | `/api/search` | Search for relevant research papers |
| POST | `/api/ask` | Generate an AI answer with retrieved sources |

### Example: Search for research papers

**Request**

```json
{
  "query": "retrieval augmented generation",
  "top_k": 5
}
```

Send this JSON to `POST /api/search`.

### Example: Ask a research question

**Request**

```json
{
  "query": "What are the main challenges of retrieval augmented generation?"
}
```

Send this JSON to `POST /api/ask`.

The search endpoint returns matching papers. The question endpoint returns an answer and source information.

## 🔐 Environment Variables

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Authenticates requests to the Gemini API |

Do not expose API keys in frontend JavaScript, public repositories, screenshots, or logs.

## 🖼️ Screenshots

Screenshots will be added after the application is tested and deployed.

Suggested screenshots:
- Dashboard
- Research paper search results
- AI-generated research answer and sources

Save screenshots in a `screenshots/` folder and link them here when available.

## 🚀 Deployment

The application can be prepared for cloud deployment after configuring the Gemini API key, dependencies, and production CORS settings.

The FastAPI service and Streamlit application are separate entry points. The standalone HTML frontend also requires a reachable backend API URL. Configure the deployed frontend and backend URLs before publishing the public application.

## 🔮 Future Improvements

- Expand and refresh the research-paper dataset.
- Improve retrieval quality and evaluate answer relevance.
- Add persistent user libraries and saved-paper storage.
- Add automated tests and production monitoring.
- Improve deployment configuration and API security.

## 👨‍💻 Author

**Nitesh Kumawat**

- GitHub: [nitukumawat80-star](https://github.com/nitukumawat80-star)
- Project repository: [PaperPilot AI](https://github.com/nitukumawat80-star/PaperPilot-AI)

---

*PaperPilot AI — Discover research. Understand faster. Explore knowledge with AI.*
