"""
Dual-metric evaluation for the grounded RAG explainer: deterministic
(fact overlap, citation coverage, embedding similarity to a reference
explanation) plus an LLM-judge rubric -- so neither score is trusted
alone, the same pattern used by the Legal SLM SFT project's evaluate.py.
"""

import json
import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from processors.embedding_generator import EmbeddingGenerator
from processors.grounded_explainer import GroundedExplanation


@dataclass
class DeterministicScore:
    fact_overlap: float
    citation_coverage: float
    embedding_similarity: float

    @property
    def composite(self) -> float:
        return float(np.mean([self.fact_overlap, self.citation_coverage, self.embedding_similarity]))


@dataclass
class JudgeScore:
    faithfulness: int
    relevance: int
    citation_accuracy: int
    rationale: str = ""

    @property
    def composite(self) -> float:
        return float(np.mean([self.faithfulness, self.relevance, self.citation_accuracy]))


def score_fact_overlap(explanation_text: str, key_facts: List[str]) -> float:
    """Fraction of `key_facts` whose salient tokens all appear in the
    explanation text (case-insensitive). A crude but deterministic,
    reproducible proxy for "did the explanation actually mention this"."""
    if not key_facts:
        return 0.0

    text = explanation_text.lower()
    hits = 0
    for fact in key_facts:
        tokens = [t for t in re.findall(r"[a-z0-9]+", fact.lower()) if len(t) > 2]
        if tokens and all(token in text for token in tokens):
            hits += 1
    return hits / len(key_facts)


def score_citation_coverage(num_citations: int, num_unverified: int) -> float:
    """Fraction of citations that passed `CitationVerifier`'s substring
    check. 1.0 if there were no citations to check (nothing to fail)."""
    if num_citations == 0:
        return 1.0
    verified = num_citations - num_unverified
    return max(0.0, verified) / num_citations


def score_embedding_similarity(explanation_text: str, ground_truth: str,
                                embedding_generator: EmbeddingGenerator) -> float:
    """Cosine similarity between the generated and ground-truth explanation
    embeddings, rescaled from [-1, 1] to [0, 1]."""
    embeddings = embedding_generator.generate_embeddings([explanation_text, ground_truth])
    if embeddings.shape[0] < 2:
        return 0.0
    a, b = embeddings[0], embeddings[1]
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    cosine = float(np.dot(a, b) / denom) if denom else 0.0
    return (cosine + 1.0) / 2.0


def compute_deterministic_score(
    explanation_text: str,
    key_facts: List[str],
    grounded: Optional[GroundedExplanation],
    ground_truth: str,
    embedding_generator: EmbeddingGenerator,
) -> DeterministicScore:
    fact_overlap = score_fact_overlap(explanation_text, key_facts)
    num_citations = len(grounded.citations) if grounded else 0
    num_unverified = len(grounded.unverified_citation_markers) if grounded else 0
    citation_coverage = score_citation_coverage(num_citations, num_unverified)
    embedding_similarity = score_embedding_similarity(explanation_text, ground_truth, embedding_generator)
    return DeterministicScore(fact_overlap, citation_coverage, embedding_similarity)


class LLMJudge:
    """Groq-based rubric judge scoring faithfulness/relevance/citation
    accuracy 1-5 with structured JSON output. Reuses the client-init
    pattern from `GroqExplanationStrategy` rather than adding a new
    dependency."""

    def __init__(self, api_key: str, model: str = 'qwen/qwen3.6-27b'):
        self.api_key = api_key
        self.model = model
        self.client = None
        self._is_available = False
        try:
            from groq import Groq
            self.client = Groq(api_key=api_key)
            self._is_available = True
        except Exception:
            self._is_available = False

    def is_available(self) -> bool:
        return self._is_available

    def judge(self, event_context: str, ground_truth: str, generated_explanation: str) -> JudgeScore:
        if not self._is_available or not self.client:
            raise RuntimeError("LLM judge not available (Groq client failed to initialize)")

        prompt = f"""You are grading a financial anomaly explanation against a reference explanation.
Score the GENERATED explanation on three axes, 1 (poor) to 5 (excellent):
- faithfulness: does it avoid contradicting or fabricating facts not supported by the event context?
- relevance: does it actually address the event and its likely cause?
- citation_accuracy: where it cites sources, do the citations plausibly support the claim next to them?

Event context:
{event_context}

Reference (ground truth) explanation:
{ground_truth}

Generated explanation to grade:
{generated_explanation}

Respond with a single JSON object:
{{"faithfulness": <1-5>, "relevance": <1-5>, "citation_accuracy": <1-5>, "rationale": "<one sentence>"}}"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_completion_tokens=300,
            response_format={"type": "json_object"},
            reasoning_effort="none",
        )
        data = json.loads(response.choices[0].message.content)
        return JudgeScore(
            faithfulness=int(data.get("faithfulness", 1)),
            relevance=int(data.get("relevance", 1)),
            citation_accuracy=int(data.get("citation_accuracy", 1)),
            rationale=data.get("rationale", ""),
        )


# --- Grounding metrics for the real-event / baseline-comparison harness
# (evaluate_baselines.py). Named precisely, since "coverage" in
# DeterministicScore above actually measures citation *precision*
# (verified / total citations) -- kept as-is there for backward
# compatibility with the synthetic eval harness, but the three metrics
# below use the more exact terminology.

CLAIM_TRIGGER_RE = re.compile(
    r"\bsimilar to\b|\bconsistent with\b|\bunlike\b|\bcompared to\b|\bcompares to\b|"
    r"\bas (?:in|with)\b|\bpreviously\b|\bprior\b|\blast time\b|\bhistorical(?:ly)?\b|"
    r"\bbefore\b|\bagain\b|\brecurring\b|\bsame pattern\b|\bsame as\b|\becho(?:es|ed)?\b|"
    r"\bmirrors?\b|\bparallel(?:s)?\b|\breminiscent\b|"
    r"\b(?:19|20)\d{2}\b|"  # a 4-digit year
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b",
    re.IGNORECASE,
)
CITATION_MARKER_RE = re.compile(r"\[S\d+\]")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]


def score_citation_precision(num_citations: int, num_unverified: int) -> float:
    """Of the citations the model actually produced, what fraction did
    `CitationVerifier` confirm are a real substring of their cited
    source? 1.0 if it produced no citations (nothing to be wrong about)."""
    if num_citations == 0:
        return 1.0
    return max(0.0, num_citations - num_unverified) / num_citations


def score_claim_grounding(explanation_text: str) -> dict:
    """Heuristic grounding-completeness check: of the sentences that
    plausibly make a historical/comparative claim (regex trigger match --
    'similar to', a bare year, a month name, etc.), what fraction actually
    carry an inline [Sx] citation marker? This is a deterministic,
    reproducible proxy, not a semantic claim detector -- a sentence can
    trip the regex without truly needing a citation, or make an uncited
    claim the regex misses. Returns citation_coverage (completeness) and
    unsupported_claim_rate (its complement) together since they're
    computed from the same sentence split."""
    sentences = split_sentences(explanation_text)
    claim_sentences = [s for s in sentences if CLAIM_TRIGGER_RE.search(s)]

    if not claim_sentences:
        return {"citation_coverage": 1.0, "unsupported_claim_rate": 0.0, "num_claim_sentences": 0}

    cited = [s for s in claim_sentences if CITATION_MARKER_RE.search(s)]
    coverage = len(cited) / len(claim_sentences)
    return {
        "citation_coverage": coverage,
        "unsupported_claim_rate": 1.0 - coverage,
        "num_claim_sentences": len(claim_sentences),
    }
