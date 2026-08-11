# Project Log: Financial Anomaly Detection Using RAG

A running record of the ideation, decisions, dead ends, bugs, and fixes
behind this project -- not a changelog of *what* changed (git history has
that), but *why*, and what we learned along the way.

---

## 1. Where this started

`F.A.I.L_OOPS` ("Financial Anomaly Interpretability using LLMs") was a
4-person MA5750 (Object-Oriented Programming) course project: Z-score/
Isolation-Forest anomaly detection over market returns, Finnhub/Yahoo news
lookup, whole-event sentence-transformer embeddings, FAISS top-k similarity,
and an LLM (Groq) explanation of the anomaly. Real shared authorship (`git
log` on the original repo: 6/5/1/1 commits across four people) -- so per the
user's own `todo.md`, the right move was a personal v2 fork, not renaming
the shared repo in place.

The existing pipeline was RAG-*adjacent* but not real RAG: no chunking, no
reranking, no grounded citations, and retrieval was over whole events, not
source text. The ask was to add real RAG as new Strategy classes behind the
existing interfaces (additive, not a rewrite), plus evaluation rigor and a
production surface (API, tests, Docker, light MLOps, an MCP tool) --
deliberately keeping the OOP pattern usage intact, since it's the actual
point of the underlying course project, while foregrounding the RAG
architecture and engineering rigor as the headline.

---

## 2. Fork hygiene took several passes to get right

The first attempt renamed the local `F.A.I.L_OOPS` directory in place and
pushed the restructure straight to a freshly created GitHub repo
(`rtrdsgpt/Financial-Anomaly-Detection-Using-RAG`). That repo's `origin`
remote was correct and the *original* team repo (renamed to `team-origin`
locally, never pushed to) was never touched -- verified after the fact via
`git ls-remote` and `gh repo view`, both confirming the team repo's GitHub
`HEAD` never moved from its last real commit.

But the user wanted zero shared history and zero risk of the old
collaborators showing up anywhere near the new repo, not just a correctly
pointed `origin`. Fix: deleted the new GitHub repo, ran `rm -rf .git && git
init` locally to drop every commit inherited from the team repo, and pushed
a single fresh `Initial commit` carrying only the current file state.
`gh repo delete` needed a `delete_repo` token scope the CLI didn't have;
worked around it once authorized rather than trying to force it.

**Lesson**: "fork, don't rename" and "no shared history at all" are two
different bars. The first protects the *other* team's repo; only the second
protects *this* repo from ever looking like it descended from the team one.
When someone says "no history," check what's actually still reachable via
`git log`/`git ls-remote`, not just where `origin` points.

---

## 3. Directory layout: flat OOP packages, then src/, reverted, then src/ again

The approved plan initially kept the existing flat `core/`/`processors/`/
`pipeline/`/`ui/` layout at the repo root, explicitly *not* restructuring
it, since the OOP package structure is the point of the course project. Mid-
build, the user asked to match the sibling `Legal SLM SFT` project's
`src/`-layout convention instead. Moved everything under `src/`
(`src/core/`, `src/processors/`, etc.) via `git mv` so history follows, and
added `pyproject.toml` with `[tool.setuptools.packages.find] where =
["src"]` -- this is the part that matters: with src-layout packaging, `core`,
`processors`, `pipeline`, `ui`, `config`, `experiments`, and `serving` all
still install as top-level packages, so every existing `from core.base
import ...`-style import kept working unchanged after `pip install -e .`.
The OOP subpackage structure itself (`core`/`processors`/`pipeline`) was
preserved as subpackages under `src/`, not flattened -- matching Legal SLM
SFT's `src/lib/` convention in spirit, but keeping this project's own
package names since they're meaningful here (Strategy/Factory/Observer
groupings), not generic.

That restructure got reverted once (back to the flat layout) during the fork-
hygiene confusion in section 2, then redone identically once the repo
situation was actually sorted out -- redoing it was cheap specifically
because it was `git mv` + a `pyproject.toml` addition, not an import
rewrite.

---

## 4. RAG design decisions

- **Headline-level chunking as the default `ChunkingStrategy`.** The
  existing `News_Headlines` field is already a `'; '`-joined string of
  distinct headlines -- that's the natural, already-present chunk boundary,
  and what actually fixes "the pipeline embeds whole events, not chunked
  text" rather than introducing an arbitrary character-count split as the
  primary strategy. `RecursiveCharacterChunker` exists alongside it for
  longer free text (filings, full articles) if that's ever added as a news
  source.
- **Chroma as the default vector store, numpy as the explicit fallback.**
  `ChromaVectorStoreStrategy` and `NumpyVectorStoreStrategy` both implement
  the same `VectorStoreStrategy` interface (an extension of the existing
  `SimilarityStrategy`, so both remain drop-in for `SimilarityAnalyzer` too)
  and both carry document text/metadata alongside vectors, which plain
  `FAISSStrategy` doesn't -- needed for citations to reference actual source
  text. Chroma's local `PersistentClient` needs no server; numpy is the
  fallback if `chromadb` isn't installed (logged clearly, not silent) and
  what the Airflow DAG's `explain_with_citations` task uses deliberately,
  since a `PersistentClient` handle doesn't cross the process boundary
  between Airflow tasks cleanly.
- **A real (if crude) hallucination guard, not a prompt instruction.**
  `CitationVerifier` substring-matches every cited quote against its
  source chunk after generation, rather than just asking the model nicely
  to cite accurately. `unverified_citation_markers` is a first-class field
  on `GroundedExplanation`, not an afterthought -- both the API response and
  the eval harness's `citation_coverage` metric read it directly.
- **Bounded adaptive retry, not open-ended agentic retry.** `RAGRetriever.
  retrieve_with_adaptive_retry` widens the historical lookback window and
  re-retrieves if the best reranked score misses a relevance floor, capped
  at `max_retries` (default 2). Deterministic and boring on purpose --
  tested against a scripted reranker (`tests/test_rag_retriever.py`) rather
  than real embeddings, since there's no controllable relevance signal to
  assert against with real model output.
- **Dual-metric eval, matching the pattern already proven in Legal SLM
  SFT's `evaluate.py`.** Deterministic (key-fact token overlap +
  citation-coverage + embedding cosine similarity to a reference
  explanation) plus an LLM-judge rubric (Groq, faithfulness/relevance/
  citation-accuracy 1-5), so neither score alone is trusted. `eval_set.json`
  is 13 hand-written, explicitly-labeled-synthetic fixtures -- fictional
  tickers/companies on purpose, so nothing in the eval set could be mistaken
  for a real historical event.
- **No fabricated results.** `evaluate.py` exits with an explicit error
  instead of producing a report if `GROQ_API_KEY` isn't configured, and the
  README's Results section says exactly that (pending a real run) rather
  than showing invented numbers -- this was an explicit constraint from the
  user during planning, not an incidental choice.

---

## 5. MLOps kept deliberately bounded

MLflow (local file-store `mlruns/`, no tracking server) logs each
`evaluate.py` run's config and composite scores. Airflow uses
`LocalExecutor` + Postgres -- no Celery/Redis/worker fleet, which would be
overkill for one daily DAG. The DAG's five tasks (`load_data` ->
`detect_anomalies` -> `retrieve_and_chunk` -> `explain_with_citations` ->
`save_results`) are thin `PythonOperator`s around the *same* Strategy
classes the API/CLI/Streamlit consumers use, with state handed between
tasks by filename via XCom (each task saves to `results/`, the next loads by
the pushed filename) rather than passing DataFrames through XCom directly,
which Airflow doesn't handle well at any real size.

---

## 6. Lint/test pass surfaced a few real, small issues

Writing the pytest suite (fully offline/mocked -- fake embedding generator,
fake reranker, fake Groq client stand-ins, no real model downloads or
network calls) and then running `ruff check src/ tests/` for the first time
surfaced:

- Two bare `except:` clauses in `news_retriever.py` (pre-existing, from the
  original team code) -- narrowed to `except Exception:`, which is strictly
  safer (a bare `except` also swallows `KeyboardInterrupt`/`SystemExit`)
  and not a behavior change for the intended "is this external service
  reachable" check.
- `main_oop.py`'s `sys.path.append(...)` runs before its `core`/`pipeline`
  imports on purpose (needed so the script works when run directly, without
  `pip install -e .` first) -- `noqa: E402`'d rather than restructured,
  since restructuring would break the direct-run case it exists for.
- `__init__.py` files across all four original packages re-export their
  submodules via `from .x import *` -- a predates-this-fork convenience
  pattern ruff can't verify is collision-free. Rather than rewriting every
  module's public surface into explicit `__all__` lists for a project this
  size, added a targeted `per-file-ignores` entry for `F403` on
  `__init__.py` in `pyproject.toml`.
- ~75 genuinely unused imports, auto-fixed via `ruff check --fix` and
  re-verified against the full test suite afterward.
