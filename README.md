# PaperPilot AI — AI-Powered Research Assistant

PaperPilot AI is an AI-powered research assistant that helps users discover relevant research papers and ask questions about research topics using AI-generated answers supported by source papers.

## 🚀 Live Demo

[Try PaperPilot AI](https://paperpilot-ai-researchpaper.streamlit.app/)

## ✨ Features

- **Research Paper Search:** Find relevant papers using search queries.
- **Hybrid Search:** Combine semantic search with keyword-based retrieval.
- **AI Question Answering:** Ask research-related questions and receive AI-generated answers.
- **Source References:** View supporting research papers alongside answers.
- **Interactive Interface:** Explore research through a Streamlit application.

## 🛠️ Tech Stack

- **Language:** Python
- **Frontend:** Streamlit
- **Semantic Search:** Sentence Transformers
- **Keyword Search:** BM25
- **AI Integration:** Google Gemini API
- **Data Processing:** Pandas, NumPy

## 📁 Project Structure

```text
PaperPilot-AI/
├── data/
│   ├── papers.csv
│   └── papers_clean.csv
├── frontend/
├── src/
│   ├── collect_papers.py
│   ├── clean_papers.py
│   ├── semantic_search.py
│   ├── hybrid_search.py
│   └── rag_pipeline.py
├── api.py
├── app.py
├── requirements.txt
└── README.md
```

## ⚙️ Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/nitukumawat80-star/PaperPilot-AI.git
   cd PaperPilot-AI
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   ```

   Windows PowerShell:

   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root:

   ```env
   GEMINI_API_KEY=your_gemini_api_key
   ```

5. Run the application:

   ```bash
   streamlit run app.py
   ```

6. Open the local URL shown in your terminal.

## 🔐 Environment Variables

Configure `GEMINI_API_KEY` using your local `.env` file or your deployment platform's secret settings. Never commit API keys to GitHub.

## 🌐 Deployment

The Streamlit application is deployed on Streamlit Community Cloud.

**Live application:** https://paperpilot-ai-researchpaper.streamlit.app/

## 👨‍💻 Author

Developed as an AI/ML project to explore research-paper discovery, hybrid retrieval, and AI-powered question answering.
