# VicPol GPT Agent

This folder contains a lightweight retrieval-augmented generation (RAG) agent for answering questions about Victoria Police Manual (VPM) policies and relevant Victorian legislation.

## Overview

1. **Corpus ingestion** – `build_index.py` loads every `.docx` policy in `../VPMs` plus any legislation exports in `../legislation_texts`, cleans the text and slices it into overlapping word chunks. Each chunk receives OpenAI embeddings (`text-embedding-3-large` by default) and is saved to `storage/vector_store.npz` with metadata in `storage/vector_store.jsonl`.
2. **Live legislation context** – `gpt_agent.legislation_fetcher.LegislationFetcher` queries `https://www.legislation.vic.gov.au/search` (or an explicit URL) for every question and adds a trimmed excerpt to the prompt. Use `sync_legislation.py` to cache full Acts locally for better retrieval quality.
3. **Question answering** – `run_agent.py` loads the vector store, finds the most similar chunks, fetches fresh legislation text, and sends a grounded prompt to `gpt-4o-mini` (configurable). Answers always cite chunk IDs/URLs so staff can trace the source material.

### Retrieval & safety guardrails

- **Definition-aware retrieval** – the index tags every chunk as a definition, rule, exception/limitation, or supporting context. When a user mentions a defined term (child, custody, etc.), the agent forces those definitions to the top of the prompt, includes cross-policy variations, and enforces minimum depth (≥2 definitions, ≥3 rules, ≥2 exceptions where available).
- **Conflict detection** – if multiple policies define a term differently, the agent surfaces each definition, switches to *informative mode*, and appends the safety warning `These policies appear to define or apply this term differently. Seek supervisor guidance.`
- **Rule + exception layout** – context is always ordered Definitions → Exceptions → Rules → Supporting context so exceptions immediately follow the relevant terminology.
- **Reasoning template** – the prompt mandates a seven-step structure (Definitions, Rules, Exceptions, Conflicts, Application, Guidance mode, Citations) and reiterates the prioritisation logic (definitions override rules, exceptions override permissions, MUST beats SHOULD).

## Setup

1. Install dependencies (you can reuse the repository `requirements.txt`):
   ```bash
   pip install -r requirements.txt
   ```
2. Provide your OpenAI key (and optionally custom models) via environment variables:
   ```bash
   export OPENAI_API_KEY=sk-...
   export OPENAI_RESPONSES_MODEL=gpt-4o-mini   # optional
   export OPENAI_EMBEDDING_MODEL=text-embedding-3-large  # optional
   export AGENT_INCLUDE_LEGISLATION=1           # optional, ingest local legislation_texts/
   ```
3. (Optional) Pull extra legislation into `legislation_texts`:
   ```bash
   python "GPT Agent/sync_legislation.py" "https://www.legislation.vic.gov.au/in-force/acts/some-act/2024"
   ```

## Building the vector store

```bash
python "GPT Agent/build_index.py" --chunk-size 450 --chunk-overlap 75
```
This scans `VPMs/` and, if `AGENT_INCLUDE_LEGISLATION=1`, also `legislation_texts/`, generates embeddings with your OpenAI key, and writes the files that the agent queries at runtime. Re-run this script whenever you add new policies or after updating the guardrail logic so the new metadata (category/definitions) is stored.

## Asking a question

```bash
python "GPT Agent/run_agent.py" "What powers apply if a youth refuses to provide ID?"
```
Optional flags:
- `--legislation-url` – force a specific legislation page to be fetched in addition to the search results.
- `--top-k` – override the number of vector hits used for context.

The script prints the model answer followed by the cited chunk IDs/paths so you can drill into the original document.

### Using the Streamlit UI

The main `app.py` now includes a **VicPol GPT Agent** mode. Launch Streamlit as usual:

```bash
streamlit run app.py
```

Select “VicPol GPT Agent” in the sidebar to ask questions with the same guardrails, view citations, and inspect the raw model JSON without leaving the policy analyser tool.

## Key modules

- `gpt_agent/config.py` – file-system paths, OpenAI model names, chunk sizes.
- `gpt_agent/doc_loader.py` – loads `.docx`, `.txt`, `.xlsx` policy/legislation files, tags each chunk with category metadata, and extracts definition terms.
- `gpt_agent/legislation_fetcher.py` – scrapes vic.gov.au search results or specific URLs.
- `gpt_agent/vector_store.py` – thin NumPy-backed vector index with cosine similarity search and category filtering.
- `gpt_agent/agent.py` – orchestrates definition-aware retrieval, prompt construction, OpenAI Responses API call, and citation/guardrail handling.
- `gpt_agent/constants.py` – shared category names, minimum retrieval depth, and safety statement.

## Notes

- Ensure the `VPMs` directory only contains documents you are allowed to embed in OpenAI systems.
- The legislation fetcher extracts the main text of the HTML page; if the portal layout changes, adjust the CSS selectors in `legislation_fetcher.py`.
- The pipeline is intentionally lightweight so you can plug it into Streamlit or other UIs—simply import `PolicyAgent` and call `answer()` from your app.
