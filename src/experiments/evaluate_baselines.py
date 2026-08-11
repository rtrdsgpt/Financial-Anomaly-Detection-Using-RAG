#!/usr/bin/env python3
"""
Baseline comparison harness: LLM-only, LLM + legacy whole-event
retrieval, RAG (no reranker), and RAG + reranker, scored on the same
real events with the same metrics -- so the RAG upgrade's actual
contribution is measurable, not asserted. The rag_no_reranker vs.
rag_with_reranker pair is also the reranker ablation.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import get_settings
from processors.ai_explainer import GroqExplanationStrategy
from processors.chunking import ChunkerFactory
from processors.embedding_generator import EmbeddingGeneratorFactory
from processors.grounded_explainer import GroundedGroqExplanationStrategy
from processors.rag_retriever import RAGRetriever
from processors.reranker import NoOpReranker, RerankerFactory
from processors.similarity_analyzer import FAISSStrategy, SimilarityAnalyzer
from processors.vector_store import VectorStoreFactory
from experiments.metrics import score_citation_precision, score_claim_grounding, score_fact_overlap


def with_retries(fn: Callable[[], Any], attempts: int = 3, base_delay: float = 5.0) -> Any:
    """Groq's free tier rate-limits; 4 configs x ~90 real events is a lot
    of calls, so transient failures get retried with backoff rather than
    counted as a real scoring failure."""
    last_error = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:
            last_error = e
            if attempt < attempts - 1:
                time.sleep(base_delay * (attempt + 1))
    raise last_error


def build_legacy_similar_events(history: pd.DataFrame, event: Dict[str, Any],
                                 embedding_generator, top_k: int = 5) -> List[Dict[str, Any]]:
    """Reproduces the original pre-RAG pipeline's retrieval: embed each
    whole historical event (Event_Type + News_Headlines), FAISS
    nearest-neighbor, return the top-k whole events (not chunks)."""
    if history.empty:
        return []

    texts = [f"{row['Event_Type']} {row['News_Headlines']}" for _, row in history.iterrows()]
    embeddings = embedding_generator.generate_embeddings(texts)

    analyzer = SimilarityAnalyzer(FAISSStrategy())
    index = analyzer.create_index(embeddings)

    query_text = f"{event.get('Event_Type', '')} {event.get('News_Headlines', '')}"
    query_embedding = embedding_generator.generate_embeddings([query_text])[0]
    _, indices = analyzer.find_similar(query_embedding, index, k=min(top_k, len(history)))

    records = history.to_dict('records')
    return [records[i] for i in indices if 0 <= i < len(records)]


def run_llm_only(case: Dict[str, Any], groq_strategy: GroqExplanationStrategy) -> Dict[str, Any]:
    explanation = with_retries(lambda: groq_strategy.explain_event(case["event"], []))
    return {"explanation": explanation, "fact_overlap": score_fact_overlap(explanation, case["key_facts"])}


def run_legacy_retrieval(case: Dict[str, Any], groq_strategy: GroqExplanationStrategy, embedding_generator) -> Dict[str, Any]:
    history = pd.DataFrame(case["historical_context"])
    similar_events = build_legacy_similar_events(history, case["event"], embedding_generator)
    explanation = with_retries(lambda: groq_strategy.explain_event(case["event"], similar_events))
    return {"explanation": explanation, "fact_overlap": score_fact_overlap(explanation, case["key_facts"])}


def run_rag(case: Dict[str, Any], grounded_strategy: GroundedGroqExplanationStrategy) -> Dict[str, Any]:
    grounded = with_retries(lambda: grounded_strategy.explain_event_grounded(case["event"], case["historical_context"]))
    explanation = grounded_strategy.render(grounded)

    result = {
        "explanation": explanation,
        "fact_overlap": score_fact_overlap(grounded.explanation, case["key_facts"]),
        "citation_precision": score_citation_precision(len(grounded.citations), len(grounded.unverified_citation_markers)),
        "num_citations": len(grounded.citations),
    }
    result.update(score_claim_grounding(grounded.explanation))
    return result


def mean(values: List[Any]) -> Any:
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare LLM-only / legacy-retrieval / RAG / RAG+reranker on real events")
    parser.add_argument("--eval-set", default=str(Path(__file__).parent / "real_eval_set.json"))
    parser.add_argument("--limit", type=int, default=None, help="cap number of cases, for a quick smoke run first")
    parser.add_argument("--config-delay", type=float, default=1.0, help="seconds between Groq calls within a case")
    parser.add_argument("--output", default="results/eval/baseline_comparison_report.json")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.groq_api_key:
        print("GROQ_API_KEY is not configured -- cannot run baseline comparisons.", file=sys.stderr)
        sys.exit(1)

    cases = json.loads(Path(args.eval_set).read_text())["cases"]
    if args.limit:
        cases = cases[:args.limit]

    embedding_generator = EmbeddingGeneratorFactory.create_generator('sentence_transformer')
    chunker = ChunkerFactory.create_chunker('headline')
    groq_strategy = GroqExplanationStrategy(settings.groq_api_key)

    noop_retriever = RAGRetriever(chunker, embedding_generator, VectorStoreFactory.create('numpy'), NoOpReranker())
    rag_noop_strategy = GroundedGroqExplanationStrategy(settings.groq_api_key, noop_retriever)

    cross_encoder = RerankerFactory.create_reranker('cross_encoder')
    if not cross_encoder.is_available():
        print("WARNING: cross-encoder unavailable -- rag_with_reranker will fall back to no-op "
              "(its numbers will equal rag_no_reranker's; the ablation won't be meaningful this run)", file=sys.stderr)
        cross_encoder = NoOpReranker()
    reranked_retriever = RAGRetriever(chunker, embedding_generator, VectorStoreFactory.create('numpy'), cross_encoder)
    rag_reranked_strategy = GroundedGroqExplanationStrategy(settings.groq_api_key, reranked_retriever)

    configs: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
        "llm_only": lambda case: run_llm_only(case, groq_strategy),
        "llm_legacy_retrieval": lambda case: run_legacy_retrieval(case, groq_strategy, embedding_generator),
        "rag_no_reranker": lambda case: run_rag(case, rag_noop_strategy),
        "rag_with_reranker": lambda case: run_rag(case, rag_reranked_strategy),
    }

    all_results: Dict[str, List[Dict[str, Any]]] = {name: [] for name in configs}
    for i, case in enumerate(cases):
        print(f"[{i + 1}/{len(cases)}] {case['id']}")
        for name, fn in configs.items():
            try:
                result = fn(case)
                result["case_id"] = case["id"]
                all_results[name].append(result)
                extra = f" citation_precision={result['citation_precision']:.2f}" if 'citation_precision' in result else ""
                print(f"  {name}: fact_overlap={result['fact_overlap']:.2f}{extra}")
            except Exception as e:
                print(f"  {name}: FAILED after retries ({e})")
                all_results[name].append({"case_id": case["id"], "error": str(e)})
            time.sleep(args.config_delay)

    summary = {}
    for name, results in all_results.items():
        ok = [r for r in results if "error" not in r]
        entry = {
            "num_cases": len(results),
            "num_successful": len(ok),
            "mean_fact_overlap": mean([r.get("fact_overlap") for r in ok]),
        }
        if any("citation_precision" in r for r in ok):
            entry["mean_citation_precision"] = mean([r.get("citation_precision") for r in ok])
            entry["mean_citation_coverage"] = mean([r.get("citation_coverage") for r in ok])
            entry["mean_unsupported_claim_rate"] = mean([r.get("unsupported_claim_rate") for r in ok])
        summary[name] = entry

    print("\n" + "=" * 60)
    print("BASELINE COMPARISON SUMMARY")
    print("=" * 60)
    for name, s in summary.items():
        print(f"\n{name}:")
        for k, v in s.items():
            print(f"  {k}: {v}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"summary": summary, "results": all_results}, indent=2, default=str))
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
