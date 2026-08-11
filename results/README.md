# results/

Generated pipeline and evaluation output -- gitignored except this file
and `eval/.gitkeep`. Populate by running the pipeline (`python
src/main_oop.py`) or the eval harness (`python src/experiments/evaluate.py`).

- `events_with_news_*.parquet` / `events_with_embeddings_*.parquet` --
  detected anomaly events for a given run.
- `explanations_*.txt` -- generated explanations for a run.
- `pipeline_summary_*.txt` -- per-run config + summary.
- `eval/eval_report_*.json` -- evaluation reports from `evaluate.py`
  (deterministic + LLM-judge scores per case).
