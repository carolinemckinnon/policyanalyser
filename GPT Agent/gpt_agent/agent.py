"""Core policy agent implementation."""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from openai import OpenAI

from .config import AgentConfig
from .embeddings import embed_query
from .legislation_fetcher import LegislationFetcher
from .text_utils import slugify
from .vector_store import RetrievedChunk, VectorStore
from .constants import (
    CATEGORY_DEFINITION,
    CATEGORY_EXCEPTION,
    CATEGORY_OTHER,
    CATEGORY_RULE,
    CATEGORY_ORDER,
    MAX_DEFINITION_ENTRIES_PER_TERM,
    MIN_DEFINITIONS,
    MIN_EXCEPTIONS,
    MIN_RULES,
    SAFETY_STATEMENT,
)


@dataclass
class AgentAnswer:
    text: str
    citations: List[Dict[str, str]]
    raw_response: dict
    conflicts: List[str]


class PolicyAgent:
    def __init__(self, config: AgentConfig, fetcher: Optional[LegislationFetcher] = None):
        self.config = config
        self.vector_store = VectorStore.load(config.vector_path, config.metadata_path)
        self.client = OpenAI()
        self.fetcher = fetcher or LegislationFetcher(max_chars=config.legislation_search_chars)
        self.definition_index, self.known_terms = self._build_definition_index()
        self.category_availability = self._compute_category_availability()
        self.doc_definitions, self.doc_titles = self._load_policy_definitions()

    def answer(
        self,
        question: str,
        legislation_url: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> AgentAnswer:
        full_query = self._compose_query_with_history(question, history)
        query_embedding = embed_query(full_query, model=self.config.embedding_model)
        sections, ordered_chunks, question_terms = self._prepare_context(question, query_embedding)
        conflicts = self._detect_definition_conflicts(sections[CATEGORY_DEFINITION])
        informative_mode, informative_reasons = self._determine_informative_mode(sections, conflicts)
        legislation_snippet = (
            self.fetcher.fetch_url(legislation_url) if legislation_url else self.fetcher.search_portal(question)
        )
        prompt = self._build_prompt(
            question=question,
            sections=sections,
            legislation_snippet=legislation_snippet,
            conflicts=conflicts,
            informative_mode=informative_mode,
            informative_reasons=informative_reasons,
            question_terms=question_terms,
            history=history or [],
        )
        response = self._generate_response(prompt)
        text = self._extract_text(response)
        text = self._replace_chunk_ids_with_titles(text, ordered_chunks)
        text = self._apply_informative_guardrail(text, informative_mode)
        definition_notes = self._collect_definition_notes(text, ordered_chunks)
        text = self._insert_definition_notes(text, definition_notes)
        citations = self._extract_citations(ordered_chunks, legislation_url, legislation_snippet)
        citations = self._filter_citations_by_usage(text, citations)
        if hasattr(response, "model_dump"):
            raw = response.model_dump()
        elif hasattr(response, "to_dict"):
            raw = response.to_dict()
        else:
            raw = {"id": getattr(response, "id", ""), "response": str(response)}
        return AgentAnswer(text=text, citations=citations, raw_response=raw, conflicts=list(conflicts.keys()))

    def _generate_response(self, prompt: str):
        messages = [
            {"role": "system", "content": "You are a precise and citation-focused VicPol policy assistant."},
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.client, "responses"):
            return self.client.responses.create(
                model=self.config.completion_model,
                temperature=self.config.temperature,
                input=messages,
            )
        return self.client.chat.completions.create(
            model=self.config.completion_model,
            temperature=self.config.temperature,
            messages=messages,
        )

    @staticmethod
    def _extract_text(response) -> str:
        outputs = getattr(response, "output", None)
        if outputs:
            parts = []
            for item in outputs:
                for content in item.content:
                    if content.type == "output_text":
                        parts.append(content.text)
            return "\n".join(parts).strip() or "No text returned."
        choices = getattr(response, "choices", None)
        if not choices:
            return "No response returned."
        texts: List[str] = []
        for choice in choices:
            message = getattr(choice, "message", None)
            if not message:
                continue
            if isinstance(message, dict):
                content = message.get("content", "")
            else:  # openai objects with attributes
                content = getattr(message, "content", "")
            if isinstance(content, list):
                texts.append(" ".join(str(part.get("text", "")) for part in content if isinstance(part, dict)))
            else:
                texts.append(str(content))
        return "\n".join(t.strip() for t in texts if t).strip() or "No text returned."

    def _extract_citations(self, chunks: List[RetrievedChunk], legislation_url: Optional[str], legislation_snippet: Optional[str]) -> List[Dict[str, str]]:
        citations: List[Dict[str, str]] = []
        for chunk in chunks:
            meta = chunk.metadata
            doc_id = meta.get("doc_id", "")
            friendly_title = self.doc_titles.get(doc_id, meta.get("title", ""))
            citations.append({
                "doc_id": doc_id,
                "chunk_id": meta.get("chunk_id", ""),
                "title": friendly_title,
                "path": meta.get("path", ""),
                "category": meta.get("category", ""),
                "source_type": meta.get("source_type", ""),
                "snippet": meta.get("text") or chunk.text,
            })
        if legislation_snippet:
            citations.append({
                "doc_id": "legislation_portal",
                "chunk_id": "legislation_portal",
                "title": legislation_url or "legislation search",
                "path": legislation_url or "https://www.legislation.vic.gov.au/search",
                "category": "LEGISLATION",
                "source_type": "Legislation",
                "snippet": legislation_snippet,
            })
        return citations

    def _build_definition_index(self) -> Tuple[Dict[str, List[dict]], Set[str]]:
        index: Dict[str, List[dict]] = defaultdict(list)
        term_set: Set[str] = set()
        for meta in self.vector_store.metadata:
            terms = meta.get("definition_terms") or []
            for term in terms:
                norm = term.strip().lower()
                if not norm:
                    continue
                index[norm].append(meta)
                term_set.add(norm)
        return index, term_set

    def _compute_category_availability(self) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for meta in self.vector_store.metadata:
            category = self._category_from_metadata(meta)
            counts[category] += 1
        return counts

    def _prepare_context(self, question: str, query_embedding) -> Tuple[Dict[str, List[RetrievedChunk]], List[RetrievedChunk], Set[str]]:
        buckets: Dict[str, List[RetrievedChunk]] = {cat: [] for cat in CATEGORY_ORDER}
        selected_ids: Set[str] = set()
        question_terms = self._detect_defined_terms(question)
        base_chunks = self.vector_store.similarity_search(query_embedding, top_k=self.config.top_k)
        for chunk in base_chunks:
            self._add_chunk_to_bucket(chunk, buckets, selected_ids)
        forced_definitions = self._fetch_definition_chunks_for_terms(question_terms, selected_ids)
        for chunk in forced_definitions:
            self._add_chunk_to_bucket(chunk, buckets, selected_ids)
        self._ensure_minimum(buckets, CATEGORY_DEFINITION, MIN_DEFINITIONS, query_embedding, selected_ids)
        self._ensure_minimum(buckets, CATEGORY_RULE, MIN_RULES, query_embedding, selected_ids)
        self._ensure_minimum(buckets, CATEGORY_EXCEPTION, MIN_EXCEPTIONS, query_embedding, selected_ids)
        ordered = [chunk for category in CATEGORY_ORDER for chunk in buckets.get(category, [])]
        return buckets, ordered, question_terms

    def _add_chunk_to_bucket(self, chunk: RetrievedChunk, buckets: Dict[str, List[RetrievedChunk]], selected_ids: Set[str]) -> None:
        chunk_id = chunk.metadata.get("chunk_id")
        if not chunk_id or chunk_id in selected_ids:
            return
        category = self._category_from_metadata(chunk.metadata)
        buckets.setdefault(category, []).append(chunk)
        selected_ids.add(chunk_id)

    def _fetch_definition_chunks_for_terms(self, terms: Set[str], selected_ids: Set[str]) -> List[RetrievedChunk]:
        forced: List[RetrievedChunk] = []
        for term in terms:
            entries = self.definition_index.get(term, [])
            for meta in entries[:MAX_DEFINITION_ENTRIES_PER_TERM]:
                chunk_id = meta.get("chunk_id")
                if not chunk_id or chunk_id in selected_ids:
                    continue
                forced.append(RetrievedChunk(metadata=meta, score=1.0))
        return forced

    def _ensure_minimum(
        self,
        buckets: Dict[str, List[RetrievedChunk]],
        category: str,
        required: int,
        query_embedding,
        selected_ids: Set[str],
    ) -> None:
        available = self.category_availability.get(category, 0)
        target = min(required, available) if available else 0
        if target == 0:
            return
        missing = target - len(buckets.get(category, []))
        if missing <= 0:
            return
        extra_chunks = self.vector_store.similarity_search(
            query_embedding,
            top_k=max(self.config.top_k * 2, target * 2),
            filter_fn=lambda meta: self._category_from_metadata(meta) == category and meta.get("chunk_id") not in selected_ids,
        )
        for chunk in extra_chunks:
            self._add_chunk_to_bucket(chunk, buckets, selected_ids)
            missing -= 1
            if missing <= 0:
                break

    def _category_from_metadata(self, metadata: dict) -> str:
        value = (metadata.get("category") or CATEGORY_RULE).upper()
        if value not in CATEGORY_ORDER:
            return CATEGORY_OTHER
        return value

    def _detect_defined_terms(self, question: str) -> Set[str]:
        normalized = question.lower()
        matches: Set[str] = set()
        for term in self.known_terms:
            if not term:
                continue
            pattern = rf"\b{re.escape(term)}\b"
            if re.search(pattern, normalized):
                matches.add(term)
        return matches

    def _detect_definition_conflicts(self, definition_chunks: List[RetrievedChunk]) -> Dict[str, List[RetrievedChunk]]:
        grouped: Dict[str, Dict[str, RetrievedChunk]] = defaultdict(dict)
        for chunk in definition_chunks:
            terms = chunk.metadata.get("definition_terms") or []
            doc_id = chunk.metadata.get("doc_id") or chunk.metadata.get("path") or chunk.metadata.get("title")
            for term in terms:
                key = term.lower()
                grouped[key][str(doc_id)] = chunk
        return {term: list(entries.values()) for term, entries in grouped.items() if len(entries) > 1}

    def _determine_informative_mode(
        self,
        sections: Dict[str, List[RetrievedChunk]],
        conflicts: Dict[str, List[RetrievedChunk]],
    ) -> Tuple[bool, List[str]]:
        reasons: List[str] = []
        if conflicts:
            reasons.append("Conflicting definitions detected.")
        if len(sections.get(CATEGORY_DEFINITION, [])) < MIN_DEFINITIONS:
            reasons.append("Insufficient definition coverage.")
        if len(sections.get(CATEGORY_RULE, [])) < MIN_RULES:
            reasons.append("Insufficient rule coverage.")
        if self.category_availability.get(CATEGORY_EXCEPTION, 0) >= MIN_EXCEPTIONS and len(sections.get(CATEGORY_EXCEPTION, [])) < MIN_EXCEPTIONS:
            reasons.append("Not enough exception/limitation passages retrieved.")
        return bool(reasons), reasons

    def _build_prompt(
        self,
        question: str,
        sections: Dict[str, List[RetrievedChunk]],
        legislation_snippet: Optional[str],
        conflicts: Dict[str, List[RetrievedChunk]],
        informative_mode: bool,
        informative_reasons: List[str],
        question_terms: Set[str],
        history: List[Dict[str, str]],
    ) -> str:
        def format_section(title: str, chunks: List[RetrievedChunk]) -> str:
            if not chunks:
                return f"### {title}\n(No passages retrieved)"
            blocks = [f"### {title}"]
            for chunk in chunks:
                meta = chunk.metadata
                block = (
                    f"[{meta.get('chunk_id')}] {meta.get('title')} ({meta.get('source_type')})\n"
                    f"Source: {meta.get('path')}\n{chunk.text}"
                )
                blocks.append(block)
            return "\n\n".join(blocks)

        context_parts = [
            format_section("Exceptions / Limitations", sections.get(CATEGORY_EXCEPTION, [])),
            format_section("Rules", sections.get(CATEGORY_RULE, [])),
            format_section("Supporting Context", sections.get(CATEGORY_OTHER, [])),
        ]
        legislation_block = legislation_snippet or "Legislation site returned no additional context."
        reasoning_template = (
            "Respond using the following structure with numbered headings:\n"
            "1. Rules (summarise mandatory directions first).\n"
            "2. Exceptions / Limitations.\n"
            "3. Application (apply definitions → rules → exceptions and highlight any conflicts if they exist)."
        )
        priority_rules = (
            "Definitions override general rules. Exceptions override permissions. Mandatory instructions (MUST) override optional guidance."
        )
        citation_rules = (
            "Inline citations must reference the policy name only (e.g., [VPM Protective services officers – duties and responsibilities]) "
            "instead of chunk IDs or generic labels."
        )
        conflict_note = "Conflicts detected for: " + ", ".join(sorted(conflicts.keys())) if conflicts else "No definition conflicts detected."
        detected_terms_note = "Detected defined terms: " + (", ".join(sorted(question_terms)) or "None matched")
        informative_instruction = (
            "If informative mode is required, do not give operational directions and include the safety statement verbatim: "
            f"'{SAFETY_STATEMENT}'."
        )
        instructions = (
            "You are helping Victoria Police staff interpret the Victoria Police Manual (VPM) and relevant legislation. "
            "Use only the provided passages. Do not invent policies."
        )
        context_block = "\n\n".join(context_parts)
        history_block = self._format_history(history)
        history_section = f"Conversation so far:\n{history_block}\n\n" if history_block else ""
        return (
            f"Instructions:\n{instructions}\n\n"
            f"{history_section}"
            f"Reasoning Template:\n{reasoning_template}\n\n"
            f"Priority Logic:\n{priority_rules}\n\n"
            f"Citation Rules:\n{citation_rules}\n\n"
            f"Conflict Status:\n{conflict_note}\n\n"
            f"Guidance Instruction:\n{informative_instruction}\n\n"
            f"{detected_terms_note}\n\n"
            f"Question:\n{question}\n\n"
            f"Policy Context:\n{context_block}\n\n"
            f"Legislation Context (vic.gov.au):\n{legislation_block}"
        )

    def _apply_informative_guardrail(self, text: str, informative_mode: bool) -> str:
        if informative_mode and SAFETY_STATEMENT not in text:
            text = text.rstrip() + "\n\n" + SAFETY_STATEMENT
        return text

    def _load_policy_definitions(self) -> Tuple[Dict[str, List[Dict[str, str]]], Dict[str, str]]:
        register_path = self.config.repo_root / "data" / "docs" / "document_register_enriched.csv"
        ontology_path = self.config.repo_root / "data" / "docs" / "policy_ontology.csv"
        if not register_path.exists() or not ontology_path.exists():
            return {}, {}
        try:
            register_df = pd.read_csv(register_path)
            register_df["register_id"] = register_df["register_id"].astype(int)
            doc_ids_by_register: Dict[int, List[str]] = defaultdict(list)
            titles_by_doc: Dict[str, str] = {}
            for _, row in register_df.iterrows():
                file_name = row.get("file_name") or row.get("relative_path") or ""
                stem = Path(str(file_name)).stem
                doc_id = f"vpm::{slugify(stem)}"
                doc_ids_by_register[int(row["register_id"])].append(doc_id)
                titles_by_doc[doc_id] = row.get("title") or stem
            ontology_df = pd.read_csv(ontology_path).fillna("")
        except Exception:  # pragma: no cover - fallback if files malformed
            return {}, {}
        definitions: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for _, row in ontology_df.iterrows():
            term = str(row.get("term") or "").strip()
            definition = str(row.get("definition") or "").strip()
            if not term or not definition:
                continue
            try:
                register_id = int(row.get("register_id", 0))
            except Exception:
                continue
            for doc_id in doc_ids_by_register.get(register_id, []):
                definitions[doc_id].append({
                    "term": term,
                    "definition": definition,
                    "source_title": titles_by_doc.get(doc_id, ""),
                })
        return definitions, titles_by_doc

    def _collect_definition_notes(self, answer_text: str, chunks: List[RetrievedChunk]) -> List[Dict[str, str]]:
        if not self.doc_definitions:
            return []
        notes: List[Dict[str, str]] = []
        seen: Set[str] = set()
        for chunk in chunks:
            doc_id = chunk.metadata.get("doc_id")
            if not doc_id:
                continue
            for entry in self.doc_definitions.get(doc_id, []):
                term = entry.get("term", "")
                if not term:
                    continue
                normalized = term.lower()
                if normalized in seen:
                    continue
                pattern = rf"\b{re.escape(term)}\b"
                if re.search(pattern, answer_text, re.IGNORECASE):
                    notes.append(entry)
                    seen.add(normalized)
        return notes

    def _insert_definition_notes(self, text: str, notes: List[Dict[str, str]]) -> str:
        if not notes:
            return text
        formatted = "\n".join(
            f"- {entry.get('term')}: {entry.get('definition')} ({entry.get('source_title') or 'VPM'})"
            for entry in notes
        )
        return text.rstrip() + "\n\nDefinitions referenced:\n" + formatted

    def _replace_chunk_ids_with_titles(self, text: str, chunks: List[RetrievedChunk]) -> str:
        for chunk in chunks:
            chunk_id = chunk.metadata.get("chunk_id")
            doc_id = chunk.metadata.get("doc_id")
            if not chunk_id or not doc_id:
                continue
            title = self.doc_titles.get(doc_id)
            if not title:
                continue
            pattern = re.compile(rf"\[{re.escape(chunk_id)}(?:\s*[-–]\s*[^\]]+)?\]")
            text = pattern.sub(f"[{title}]", text)
        return text

    def _filter_citations_by_usage(self, text: str, citations: List[Dict[str, str]]) -> List[Dict[str, str]]:
        useful: List[Dict[str, str]] = []
        seen_docs: Set[str] = set()
        normalized_text = text.lower()
        for citation in citations:
            doc_id = citation.get("doc_id") or citation.get("chunk_id")
            if doc_id in seen_docs:
                continue
            title = (citation.get("title") or "").strip()
            keep = False
            if citation.get("category") == "LEGISLATION":
                keep = True
            elif title and title.lower() in normalized_text:
                keep = True
            if keep:
                useful.append(citation)
                seen_docs.add(doc_id or title)
        return useful

    def _compose_query_with_history(self, question: str, history: Optional[List[Dict[str, str]]]) -> str:
        if not history:
            return question
        recent = history[-3:]
        history_fragments = []
        for entry in recent:
            user = entry.get("question", "")
            assistant = entry.get("answer", "")
            history_fragments.append(f"User: {user}\nAssistant: {assistant}")
        joined = "\n".join(history_fragments)
        return f"{joined}\n\nCurrent question: {question}"

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        if not history:
            return ""
        lines = []
        for idx, entry in enumerate(history[-3:], start=1):
            user = entry.get("question", "").strip()
            assistant = entry.get("answer", "").strip()
            lines.append(f"{idx}. User: {user}\n   Assistant: {assistant}")
        return "\n".join(lines)
