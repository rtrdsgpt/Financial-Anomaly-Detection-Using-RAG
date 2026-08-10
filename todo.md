# F.A.I.L_OOPS — TODO

Financial Anomaly Interpretability using LLMs — built for the MA5750 OOP course, so the
Strategy/Factory/Observer/Template Method/Builder/Repository pattern architecture is the point,
not incidental polish, and should stay front and center. On top of that OOP skeleton it's already
RAG-*adjacent* (FAISS similarity search + LLM explanation). Upgrade to true RAG using new Strategy
classes, not a rewrite. See `Project Plan.md` (Projects root) section 3.

## 0. Ownership / hygiene first
- [ ] 4-person team project — check actual contribution before claiming full ownership
- [ ] Fork into a personal `v2` repo where authorship is unambiguous
- [ ] Confirm `api_keys.txt` is gitignored before linking this repo anywhere public — it currently
      holds a plaintext-key pattern

## 1. RAG upgrade (implemented as new Strategy classes behind existing factories/interfaces)
- [ ] Add `VectorStoreRetrievalStrategy` (Qdrant/Chroma/pgvector) alongside the existing
      `NewsRetrieverFactory`/FAISS-based strategy
- [ ] Add `RerankerStrategy` for retrieved candidates
- [ ] Add a proper chunking strategy for news/filings (current pipeline embeds whole
      articles/snippets, not chunked text)
- [ ] Add grounded citations in the LLM's anomaly explanation (`AIExplainerFactory` /
      `ai_explainer.py`) — every claim should cite specific retrieved source text

## 2. Evaluation
- [ ] Build a small eval set of (anomaly, ground-truth explanation) pairs
- [ ] Score faithfulness/relevance via `ragas` or a custom LLM-judge rubric — reuse the pattern
      already proven in Legal SLM SFT's `evaluate.py` (dual metric: deterministic rubric +
      LLM-judge)

## 3. Productionization
- [ ] FastAPI wrapper around the pipeline (currently Streamlit-only via `app_oop.py`) — implement
      as a new consumer of the existing Strategy interfaces, consistent with current architecture
- [ ] Dockerfile (README currently shows an example Dockerfile snippet but no actual file exists)
- [ ] Real pytest suite — README shows example test snippets but no `tests/` directory exists;
      the Strategy pattern makes each strategy trivially unit-testable in isolation
      (`processors/data_loader.py`, `anomaly_detector.py`, `news_retriever.py`,
      `embedding_generator.py`, `similarity_analyzer.py`, `ai_explainer.py`)
- [ ] CI (GitHub Actions) running the new pytest suite

## 4. Optional differentiator
- [ ] Expose "explain this anomaly" as an MCP tool so it can be called by an external agent
      (including the new Patent Prior-Art agent's citation-verification pattern, or Claude
      Desktop directly)
