#!/usr/bin/env python3
"""
CLI: run the grounded RAG explainer over eval_set.json, score with the
dual metric (deterministic + LLM judge), write a report, and log to
MLflow.

Real numbers only -- this script does not fabricate results. If
GROQ_API_KEY isn't configured it says so and exits rather than producing
a fake report; the README's Results section only shows numbers that were
actually produced by running this.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ on path when run directly

from config.settings import get_settings
from processors.chunking import ChunkerFactory
from processors.embedding_generator import EmbeddingGeneratorFactory
from processors.vector_store import VectorStoreFactory
from processors.reranker import RerankerFactory
from processors.rag_retriever import RAGRetriever
from processors.grounded_explainer import GroundedGroqExplanationStrategy
from experiments.metrics import compute_deterministic_score, LLMJudge


def load_eval_set(path: Path) -> list:
    return json.loads(path.read_text())["cases"]


def build_retriever(vector_store_backend: str, chunking_strategy: str, use_reranker: bool):
    chunker = ChunkerFactory.create_chunker(chunking_strategy)
    embedding_generator = EmbeddingGeneratorFactory.create_generator('sentence_transformer')

    if vector_store_backend == 'chroma':
        try:
            import chromadb  # noqa: F401
            vector_store = VectorStoreFactory.create('chroma')
        except ImportError:
            print("chromadb not installed -- falling back to the numpy vector store", file=sys.stderr)
            vector_store = VectorStoreFactory.create('numpy')
    else:
        vector_store = VectorStoreFactory.create(vector_store_backend)

    if use_reranker:
        reranker = RerankerFactory.create_reranker('cross_encoder')
        if not reranker.is_available():
            print("cross-encoder reranker unavailable -- falling back to no-op reranker", file=sys.stderr)
            reranker = RerankerFactory.create_reranker('none')
    else:
        reranker = RerankerFactory.create_reranker('none')

    return RAGRetriever(chunker, embedding_generator, vector_store, reranker), embedding_generator


def run_case(case: dict, strategy: GroundedGroqExplanationStrategy, embedding_generator, judge) -> dict:
    event = case["event"]
    history = case["historical_context"]

    grounded = strategy.explain_event_grounded(event, history)
    explanation_text = strategy.render(grounded)

    det = compute_deterministic_score(
        explanation_text=grounded.explanation,
        key_facts=case["key_facts"],
        grounded=grounded,
        ground_truth=case["ground_truth_explanation"],
        embedding_generator=embedding_generator,
    )

    judge_score = None
    if judge and judge.is_available():
        event_context = f"{event.get('Event_Type')}: {event.get('News_Headlines')}"
        judge_score = judge.judge(event_context, case["ground_truth_explanation"], grounded.explanation)

    return {
        "id": case["id"],
        "scenario_type": case["scenario_type"],
        "generated_explanation": explanation_text,
        "num_citations": len(grounded.citations),
        "num_unverified_citations": len(grounded.unverified_citation_markers),
        "deterministic": {
            "fact_overlap": det.fact_overlap,
            "citation_coverage": det.citation_coverage,
            "embedding_similarity": det.embedding_similarity,
            "composite": det.composite,
        },
        "judge": None if judge_score is None else {
            "faithfulness": judge_score.faithfulness,
            "relevance": judge_score.relevance,
            "citation_accuracy": judge_score.citation_accuracy,
            "composite": judge_score.composite,
            "rationale": judge_score.rationale,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the grounded RAG explainer against eval_set.json")
    parser.add_argument("--eval-set", default=str(Path(__file__).parent / "eval_set.json"))
    parser.add_argument("--vector-store-backend", default=None)
    parser.add_argument("--chunking-strategy", default=None)
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--no-judge", action="store_true", help="Skip the LLM-judge metric (deterministic only)")
    parser.add_argument("--output-dir", default="results/eval")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.groq_api_key:
        print(
            "GROQ_API_KEY is not configured (env var / .env / api_keys.txt) -- cannot run "
            "the grounded explainer against real Groq calls. Set it and re-run.",
            file=sys.stderr,
        )
        sys.exit(1)

    vector_store_backend = args.vector_store_backend or settings.vector_store_backend
    chunking_strategy = args.chunking_strategy or settings.chunking_strategy
    use_reranker = not args.no_reranker and settings.use_reranker

    retriever, embedding_generator = build_retriever(vector_store_backend, chunking_strategy, use_reranker)
    strategy = GroundedGroqExplanationStrategy(settings.groq_api_key, retriever)

    judge = None
    if not args.no_judge:
        judge = LLMJudge(settings.groq_api_key)
        if not judge.is_available():
            print("LLM judge unavailable (Groq client init failed) -- continuing with deterministic scores only",
                  file=sys.stderr)
            judge = None

    cases = load_eval_set(Path(args.eval_set))
    print(f"Running {len(cases)} eval cases (vector_store={vector_store_backend}, "
          f"chunking={chunking_strategy}, reranker={'on' if use_reranker else 'off'}, "
          f"judge={'on' if judge else 'off'})...")

    results = []
    for case in cases:
        print(f"  - {case['id']} ({case['scenario_type']})...", end=" ", flush=True)
        try:
            result = run_case(case, strategy, embedding_generator, judge)
            results.append(result)
            judge_part = f" judge={result['judge']['composite']:.2f}" if result["judge"] else ""
            print(f"deterministic={result['deterministic']['composite']:.3f}{judge_part}")
        except Exception as e:
            print(f"FAILED ({e})")
            results.append({"id": case["id"], "scenario_type": case["scenario_type"], "error": str(e)})

    successful = [r for r in results if "error" not in r]
    judged = [r for r in successful if r.get("judge")]
    summary = {
        "num_cases": len(cases),
        "num_successful": len(successful),
        "num_failed": len(cases) - len(successful),
        "mean_deterministic_composite": (
            sum(r["deterministic"]["composite"] for r in successful) / len(successful) if successful else None
        ),
        "mean_judge_composite": (
            sum(r["judge"]["composite"] for r in judged) / len(judged) if judged else None
        ),
    }

    report = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "vector_store_backend": vector_store_backend,
            "chunking_strategy": chunking_strategy,
            "use_reranker": use_reranker,
            "judge_enabled": judge is not None,
        },
        "summary": summary,
        "results": results,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"eval_report_{timestamp}.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))

    print("\n" + "=" * 60)
    print("EVAL SUMMARY")
    print("=" * 60)
    print(f"Cases: {summary['num_successful']}/{summary['num_cases']} succeeded")
    if summary["mean_deterministic_composite"] is not None:
        print(f"Mean deterministic composite: {summary['mean_deterministic_composite']:.3f}")
    if summary["mean_judge_composite"] is not None:
        print(f"Mean judge composite (1-5):   {summary['mean_judge_composite']:.2f}")
    print(f"Report written to: {report_path}")

    if not args.no_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment("financial-anomaly-rag-eval")
            with mlflow.start_run(run_name=f"eval_{timestamp}"):
                mlflow.log_params(report["config"])
                if summary["mean_deterministic_composite"] is not None:
                    mlflow.log_metric("mean_deterministic_composite", summary["mean_deterministic_composite"])
                if summary["mean_judge_composite"] is not None:
                    mlflow.log_metric("mean_judge_composite", summary["mean_judge_composite"])
                mlflow.log_artifact(str(report_path))
            print(f"Logged to MLflow (tracking_uri={settings.mlflow_tracking_uri})")
        except ImportError:
            print("mlflow not installed -- skipping MLflow logging (report was still written to disk)",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
