# VicPol Policy Tool

This repository hosts the Victoria Police policy analysis and Q&A tool. It now combines the original Streamlit policy analyser with a conversational VicPol GPT Agent that performs definition-aware retrieval across the Victoria Police Manual (VPM) and pulls live context from legislation.vic.gov.au.

## Key components

- `app.py` – Streamlit interface with two modes: the legacy policy analyser and the VicPol GPT Agent chat experience (multi-turn, citation-rich responses).
- `GPT Agent/` – self-contained RAG pipeline (document loaders, vector store builder, OpenAI agent, legislation fetcher, CLI utilities).
- `data/docs/` – enriched register CSV, policy ontology, and supporting files used for metadata and definitions.
- `requirements.txt` – Python dependencies.

## Quick start

1. Create/activate a Python 3.10+ virtual environment and install requirements:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Export your OpenAI key (and optional model overrides):
   ```bash
   export OPENAI_API_KEY=sk-...
   ```
3. Build the VPM vector store:
   ```bash
   python "GPT Agent/build_index.py"
   ```
4. Launch the app:
   ```bash
   streamlit run app.py
   ```

Switch the sidebar toggle to “VicPol GPT Agent” to start a chat. Context, definitions, and citations are shown inline with links back to the original VPM docs.

## CLI utilities

- `python "GPT Agent/run_agent.py" "Your question"`
- `python "GPT Agent/build_index.py"` (rebuild embeddings)
- `python "GPT Agent/sync_legislation.py" <legislation-url>`

## Repository status

This project is under active development; see commit history for recent changes (conversation mode, citation cleanup, guardrail updates). Contributions welcome via PR.
