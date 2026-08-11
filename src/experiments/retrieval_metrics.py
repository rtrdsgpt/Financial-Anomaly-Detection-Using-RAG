#!/usr/bin/env python3
"""
Retrieval-only metrics (Recall@k, MRR@k) and reranker lift, computed
against real_eval_set.json's real events -- no LLM calls, so this runs
without GROQ_API_KEY.

Relevance proxy: since there's no human relevance judgment for these
real events, a candidate chunk is treated as "relevant" to a query event
if it comes from a historical event for the SAME ticker with the SAME
Event_Type as the query (excluding the query's own event). This is a
simple, fully deterministic, one-sentence-explainable proxy for "did
retrieval surface a genuinely analogous historical anomaly" -- not a
human-annotated gold standard. Reported alongside how many queries had
zero relevant documents under this proxy (those are excluded from the
mean, not scored as 0, since recall/MRR are undefined without a
relevant set).
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from processors.chunking import ChunkerFactory
from processors.embedding_generator import EmbeddingGeneratorFactory
from processors.reranker import RerankerFactory
from processors.vector_store import VectorStoreFactory


def recall_at_k(retrieved_event_indices: List[Any], relevant_event_indices: set, k: int) -> Optional[float]:
    if not relevant_event_indices:
        return None
    top_k = set(retrieved_event_indices[:k])
    return len(top_k & relevant_event_indices) / len(relevant_event_indices)


def mrr_at_k(retrieved_event_indices: List[Any], relevant_event_indices: set, k: int) -> Optional[float]:
    if not relevant_event_indices:
        return None
    for rank, event_index in enumerate(retrieved_event_indices[:k], start=1):
        if event_index in relevant_event_indices:
            return 1.0 / rank
    return 0.0


@dataclass
class QueryResult:
    case_id: str
    num_relevant: int
    pre_rerank_recall5: Optional[float]
    pre_rerank_mrr5: Optional[float]
    post_rerank_recall5: Optional[float]
    post_rerank_mrr5: Optional[float]


def evaluate_case(case: Dict[str, Any], chunker, embedding_generator, vector_store_backend: str,
                   reranker, top_n: int = 10, top_k: int = 5) -> Optional[QueryResult]:
    event = case["event"]
    history = pd.DataFrame(case["historical_context"])
    if history.empty:
        return None

    # Fresh vector store per query -- these are cheap (numpy) and this
    # keeps each query's corpus isolated and reproducible.
    vector_store = VectorStoreFactory.create(vector_store_backend)

    corpus: List[Dict[str, Any]] = []
    for event_index, row in history.iterrows():
        for chunk in chunker.chunk(row.get('News_Headlines', ''), {
            'event_index': event_index, 'Event_Type': row.get('Event_Type'),
        }):
            corpus.append({'text': chunk.text, 'metadata': chunk.metadata})

    if not corpus:
        return None

    texts = [c['text'] for c in corpus]
    embeddings = embedding_generator.generate_embeddings(texts)
    vector_store.set_documents(texts, [c['metadata'] for c in corpus])
    index = vector_store.create_index(embeddings)

    relevant_event_indices = {
        i for i, row in history.iterrows()
        if row.get('Event_Type') == event.get('Event_Type')
    }
    if not relevant_event_indices:
        return None

    query_text = f"{event.get('Event_Type', '')} {event.get('News_Headlines', '')}"
    query_embedding = embedding_generator.generate_embeddings([query_text])[0]

    top_n_actual = min(top_n, len(corpus))
    _, indices = vector_store.find_similar(query_embedding, index, k=top_n_actual)
    pre_rerank_candidates = [corpus[i] for i in indices if 0 <= i < len(corpus)]
    pre_rerank_event_order = [c['metadata']['event_index'] for c in pre_rerank_candidates]

    from processors.chunking import Chunk
    rerank_input = [Chunk(text=c['text'], metadata=c['metadata']) for c in pre_rerank_candidates]
    reranked = reranker.rerank(query_text, rerank_input, top_k=len(rerank_input))
    post_rerank_event_order = [r.chunk.metadata['event_index'] for r in reranked]

    return QueryResult(
        case_id=case["id"],
        num_relevant=len(relevant_event_indices),
        pre_rerank_recall5=recall_at_k(pre_rerank_event_order, relevant_event_indices, top_k),
        pre_rerank_mrr5=mrr_at_k(pre_rerank_event_order, relevant_event_indices, top_k),
        post_rerank_recall5=recall_at_k(post_rerank_event_order, relevant_event_indices, top_k),
        post_rerank_mrr5=mrr_at_k(post_rerank_event_order, relevant_event_indices, top_k),
    )


def mean(values: List[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Recall@5/MRR@5 and reranker lift on real_eval_set.json")
    parser.add_argument("--eval-set", default=str(Path(__file__).parent / "real_eval_set.json"))
    parser.add_argument("--vector-store-backend", default="numpy")
    parser.add_argument("--output", default="results/eval/retrieval_metrics_report.json")
    args = parser.parse_args()

    data = json.loads(Path(args.eval_set).read_text())
    cases = data["cases"]

    chunker = ChunkerFactory.create_chunker("headline")
    embedding_generator = EmbeddingGeneratorFactory.create_generator('sentence_transformer')
    reranker = RerankerFactory.create_reranker('cross_encoder')
    if not reranker.is_available():
        print("WARNING: cross-encoder reranker unavailable; post-rerank metrics will equal pre-rerank.", file=sys.stderr)

    results: List[QueryResult] = []
    excluded = 0
    for case in cases:
        result = evaluate_case(case, chunker, embedding_generator, args.vector_store_backend, reranker)
        if result is None:
            excluded += 1
            continue
        results.append(result)
        print(f"{case['id']}: relevant={result.num_relevant} "
              f"pre[R@5={result.pre_rerank_recall5:.2f} MRR@5={result.pre_rerank_mrr5:.2f}] "
              f"post[R@5={result.post_rerank_recall5:.2f} MRR@5={result.post_rerank_mrr5:.2f}]")

    summary = {
        "num_cases": len(cases),
        "num_scored": len(results),
        "num_excluded_no_relevant_docs": excluded,
        "pre_rerank_mean_recall5": mean([r.pre_rerank_recall5 for r in results]),
        "pre_rerank_mean_mrr5": mean([r.pre_rerank_mrr5 for r in results]),
        "post_rerank_mean_recall5": mean([r.post_rerank_recall5 for r in results]),
        "post_rerank_mean_mrr5": mean([r.post_rerank_mrr5 for r in results]),
    }
    if summary["pre_rerank_mean_recall5"] is not None and summary["post_rerank_mean_recall5"] is not None:
        summary["reranker_recall5_lift"] = summary["post_rerank_mean_recall5"] - summary["pre_rerank_mean_recall5"]
    if summary["pre_rerank_mean_mrr5"] is not None and summary["post_rerank_mean_mrr5"] is not None:
        summary["reranker_mrr5_lift"] = summary["post_rerank_mean_mrr5"] - summary["pre_rerank_mean_mrr5"]

    print("\n" + "=" * 60)
    print("RETRIEVAL METRICS SUMMARY")
    print("=" * 60)
    for key, value in summary.items():
        print(f"{key}: {value}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "summary": summary,
        "relevance_proxy": "same ticker, same Event_Type as the query event, excluding the query itself",
        "per_case": [r.__dict__ for r in results],
    }, indent=2))
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
