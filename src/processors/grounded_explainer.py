"""
Grounded AI explanation processor implementing Strategy pattern.

Extends `AIExplanationStrategy` (`processors/ai_explainer.py`) with a
retrieval-grounded variant whose prompt requires inline `[S1]`-style
citations tied to `RAGRetriever` output, returns structured JSON, and is
then run through a deterministic `CitationVerifier` that checks each cited
quote actually substring-matches its source chunk -- a real hallucination
guard, not just a prompt instruction the model may or may not follow.
"""

import json
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from core.exceptions import AIExplanationError, APIClientError
from processors.ai_explainer import AIExplanationStrategy
from processors.reranker import RankedChunk
from processors.rag_retriever import RAGRetriever


class Citation(BaseModel):
    """One inline citation tying a claim to a specific retrieved chunk"""

    marker: str = Field(description="Inline marker used in the explanation text, e.g. 'S1'")
    source_text: str = Field(description="The exact quoted substring from the cited source chunk")
    event_index: Optional[Any] = Field(default=None, description="Index of the source event, if known")


class GroundedExplanation(BaseModel):
    """A grounded explanation plus its citations and verification outcome"""

    explanation: str
    citations: List[Citation] = Field(default_factory=list)
    unverified_citation_markers: List[str] = Field(default_factory=list)

    @property
    def is_fully_grounded(self) -> bool:
        return len(self.unverified_citation_markers) == 0


class CitationVerifier:
    """Deterministically checks each cited quote actually substring-matches
    its source chunk text, rather than trusting the LLM's citation to be
    accurate just because it produced one."""

    def verify(self, explanation: GroundedExplanation, source_chunks: List[RankedChunk]) -> GroundedExplanation:
        source_texts = [ranked.chunk.text for ranked in source_chunks]

        unverified = []
        for citation in explanation.citations:
            quote = citation.source_text.strip()
            if not quote or not any(quote in source for source in source_texts):
                unverified.append(citation.marker)

        explanation.unverified_citation_markers = unverified
        return explanation


class GroundedGroqExplanationStrategy(AIExplanationStrategy):
    """Strategy for generating citation-grounded explanations using Groq,
    with retrieval supplied by a `RAGRetriever` instead of the raw
    `similar_events` list."""

    def __init__(
        self,
        api_key: str,
        retriever: RAGRetriever,
        model: str = 'qwen/qwen3.6-27b',
        top_k_sources: int = 5,
    ):
        self.api_key = api_key
        self.retriever = retriever
        self.model = model
        self.top_k_sources = top_k_sources
        self.client = None
        self._is_available = False
        try:
            self._initialize_client()
        except Exception:
            pass

    def _initialize_client(self) -> None:
        """Initialize the Groq client (rotates across a comma-separated
        list of keys in self.api_key if one hits a rate limit)"""
        try:
            from processors.groq_key_rotation import RotatingGroqClient, parse_api_keys
            self.client = RotatingGroqClient(parse_api_keys(self.api_key))
            self._is_available = True
        except Exception as e:
            self._is_available = False
            raise APIClientError(f"Failed to initialize Groq client: {e}")

    def explain_event(self, event_data: Dict[str, Any], similar_events: List[Dict[str, Any]]) -> str:
        """Satisfies the `AIExplanationStrategy` interface by rendering the
        grounded explanation (with any unverified citations flagged inline)
        as plain text."""
        grounded = self.explain_event_grounded(event_data, similar_events)
        return self.render(grounded)

    def explain_event_grounded(self, event_data: Dict[str, Any], similar_events: List[Dict[str, Any]]) -> GroundedExplanation:
        """Retrieve source chunks via the RAG retriever, prompt for a
        structured, cited explanation, then verify each citation."""
        try:
            if not self._is_available:
                self._initialize_client()
            if not self.client:
                raise APIClientError("Groq client not initialized")

            import pandas as pd
            history = pd.DataFrame(similar_events) if similar_events else pd.DataFrame()
            query_text = f"{event_data.get('Event_Type', '')} {event_data.get('News_Headlines', '')}".strip()

            if history.empty or 'News_Headlines' not in history.columns:
                sources: List[RankedChunk] = []
            else:
                result = self.retriever.retrieve(query_text, history, top_n=max(10, self.top_k_sources * 2), top_k=self.top_k_sources)
                sources = result.candidates

            prompt = self._create_grounded_prompt(event_data, sources)

            response = self.client.create_chat_completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_completion_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
                response_format={"type": "json_object"},
                # qwen3.6-27b supports fully disabling reasoning (unlike
                # Groq's gpt-oss models, which only go down to 'low' and
                # still spend part of the token budget on hidden thinking
                # -- risking an empty response on tighter budgets like this).
                reasoning_effort="none",
            )

            raw = response.choices[0].message.content
            grounded = self._parse_response(raw)

            return CitationVerifier().verify(grounded, sources)

        except AIExplanationError:
            raise
        except Exception as e:
            raise APIClientError(f"Failed to generate grounded Groq explanation: {e}")

    def _create_grounded_prompt(self, event_data: Dict[str, Any], sources: List[RankedChunk]) -> str:
        if sources:
            source_lines = [
                f"[S{i + 1}] {ranked.chunk.text} "
                f"(Date: {ranked.chunk.metadata.get('Date')}, relevance={ranked.score:.3f})"
                for i, ranked in enumerate(sources)
            ]
            source_block = "\n".join(source_lines)
        else:
            source_block = "(no retrieved sources -- explain from the event details alone)"

        return f"""You are a financial analyst assistant. Explain the market anomaly below using
ONLY the numbered sources provided plus the event details. Every factual claim about
similar historical events must cite a source marker like [S1], [S2] inline in the
explanation text. Do not cite a source number that isn't listed below.

Sources:
{source_block}

Today's Event:
- Date: {event_data.get('Date')}
- Event Type: {event_data.get('Event_Type')}
- Z-score: {event_data.get('Z_score')}
- Return: {event_data.get('Return')}
- News: {event_data.get('News_Headlines')}

Respond with a single JSON object matching this schema exactly:
{{
  "explanation": "<explanation text with inline [S1]-style citation markers>",
  "citations": [
    {{"marker": "S1", "source_text": "<exact quoted substring from source S1>", "event_index": null}}
  ]
}}
Only include a citation entry for a marker if you actually used it in the explanation text.
The "source_text" for each citation must be an exact, verbatim substring of that source's text above."""

    def _parse_response(self, raw: str) -> GroundedExplanation:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            match = re.search(r'\{.*\}', raw or '', re.DOTALL)
            data = json.loads(match.group(0)) if match else {"explanation": raw or "", "citations": []}

        return GroundedExplanation(
            explanation=data.get("explanation", ""),
            citations=[Citation(**c) for c in data.get("citations", [])],
        )

    def render(self, grounded: GroundedExplanation) -> str:
        """Render a GroundedExplanation as plain text, flagging any
        citations that failed verification. Public so callers (e.g. the
        eval harness) that already have a `GroundedExplanation` from
        `explain_event_grounded` don't need to re-run `explain_event`."""
        text = grounded.explanation
        if grounded.unverified_citation_markers:
            flagged = ", ".join(grounded.unverified_citation_markers)
            text += f"\n\n[unverified citations: {flagged} -- could not be substring-matched to a retrieved source]"
        return text

    def is_available(self) -> bool:
        return self._is_available
