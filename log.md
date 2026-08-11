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

---

## 7. Making the eval CV-defensible: real data, real baselines, real numbers

The synthetic eval set (section on RAG design decisions above) is fine for
exercising the pipeline, but it isn't evidence of anything for a resume --
13 hand-written scenarios with hand-written key facts prove the code runs,
not that the RAG upgrade helps. The user's own priority-ordered list for
what to fix before this goes on a CV: (1) actually run an eval, (2) add
baseline comparisons (LLM-only / LLM+legacy-retrieval / RAG / RAG+reranker),
(3) real financial events (50-100+), (4) retrieval metrics (Recall@5, MRR@5,
reranker lift) measured separately from generation, (5) grounding metrics
(citation precision, unsupported-claim rate, citation coverage), (6) one
real ablation, (7, deferred) SEC filings/earnings transcripts as a second
retrieval source.

**Ground truth for real events, decided up front**: no hand-written or
LLM-written reference explanation for the 112 real events. Writing 100
references by hand isn't realistic, and an LLM-written reference would
make the embedding-similarity-to-reference metric partly circular (grading
one language model's explanation against another's explanation of the same
event). `key_facts` are instead deterministically extracted from each
event's own real headline (regex for percentages/dollar amounts + a fixed
action-word vocabulary) -- `build_real_eval_set.py`.

**Two false starts on historical news, both instructive.** First assumed
Yahoo's free RSS feed (already used elsewhere in this project) would work
for historical dates -- it doesn't; direct testing confirmed it only
returns news from roughly the last few days, nothing for a 2023 date.
Second assumed Finnhub's `/company-news` free tier, which does accept
`from`/`to` date parameters, would cover multiple years -- also wrong.
A raw request to the actual endpoint for 2023/2024 date ranges came back
`200 OK` with zero articles, while a recent window returned 250. Binary-
searched the real cutoff by testing windows 0-12 months back: real data at
11.5 months, empty at exactly 12 months. The lesson isn't "Finnhub is
broken" -- the API honestly answers whatever's asked of it with `200` and
an empty array either way, so *accepting a parameter* is not the same as
*having data for it*, and the only way to know the difference was to
actually query it, not read documentation or assume. Real event sourcing
was rebuilt around a ~1-year window ending "now" instead of the originally
planned multi-year 2021-2024 range, and still produced 112 events (target
was 50-100) across 10 tickers spanning tech/auto/pharma/energy/financials/
media/industrials -- because a year of real anomaly-adjacent news across
10 tickers turned out to be plenty, once the actual constraint was known.

**Retrieval metrics needed an honest relevance proxy.** There's no human
relevance judgment for 112 real events, and fabricating one would violate
the whole point of this exercise. Went with the simplest defensible
definition: a candidate chunk is "relevant" if it's from a historical event
for the *same ticker* with the *same `Event_Type`* as the query, excluding
the query itself -- one sentence to explain, fully deterministic,
reproducible from the code. Real result:  Recall@5 dropped slightly with
reranking (0.421 -> 0.405, lift -0.017) while MRR@5 improved slightly
(0.666 -> 0.686, lift +0.020) -- a small, mixed effect, not the clean
"reranker improved relevance by X%" story that would look best on a CV.
Reported as measured. 106/112 events were scored; 6 were excluded (not
scored as zero) because the proxy found no relevant docs for them at all --
recall/MRR are undefined without a relevant set, so silently scoring those
as 0 would have deflated the real number for a reason that has nothing to
do with retrieval quality.

**The default Groq model had been deprecated without anyone noticing.**
The very first real (non-mocked) Groq call in this project's history
returned `404 model_not_found` for `meta-llama/llama-4-scout-17b-16e-
instruct` -- the hardcoded default since the original team project. Queried
`client.models.list()` against the real account to see what's actually
available rather than guessing a replacement. Tried `openai/gpt-oss-120b`
first (seemed like the natural upgrade); it cannot fully disable reasoning
on Groq (`reasoning_effort` only goes down to `'low'`, no `'none'`), and on
a live call it spent its entire `max_completion_tokens` budget on hidden
reasoning and returned an empty response (`json_validate_failed`) --
reproduced directly, not inferred from docs. `qwen/qwen3.6-27b` does
support `reasoning_effort='none'` on Groq, confirmed with a live JSON-mode
call before wiring it in everywhere as the new default, with
`reasoning_effort="none"` set explicitly on every call site.

**A single API key wasn't enough to finish a ~450-call run.** The first
attempt at the full 112-event x 4-config baseline comparison (448 base
Groq calls, more with retries) visibly stalled: after ~55 minutes, the
Groq console showed only ~150 requests against that key, and CPU time on
the process was under 10 seconds -- almost all wall-clock time was retry
backoff, not real work. Rather than working around this ad hoc, ported the
key-rotation pattern already proven in the Exporter Crawl project
(`build_key_rotator` / `HardRateLimitError` there): `GROQ_API_KEY` now
accepts a comma-separated list, and `RotatingGroqClient`
(`processors/groq_key_rotation.py`) transparently rotates to the next key
and retries the same call when the current one hits a rate limit, rather
than the functional `judge_fn`/`next_judge_fn` closures Exporter Crawl
uses -- adapted to a client wrapper since this project's Strategy classes
call `self.client.chat.completions.create(...)` directly. A single key
still behaves exactly as before; rotation is a no-op superset.

**Status at the point this log entry was written**: real event sourcing
and retrieval metrics are complete and their numbers are in the README.
The full baseline comparison (`evaluate_baselines.py` across all 112
events x 4 configs) was still running/pending a rerun with the rotating
client -- its numbers, once produced by an actual run, go in the README's
Results section the same way retrieval's did, not before.
