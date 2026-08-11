#!/usr/bin/env python3
"""
Build a real-event evaluation set: real tickers, real Z-score anomalies
(via yfinance), real contemporaneous news headlines (via Finnhub's
company-news endpoint, which -- unlike Yahoo's free RSS feed, which
only surfaces the last few days -- supports real historical date-ranged
lookups), with `key_facts` auto-extracted deterministically
from each event's own headline text -- not hand-written, not LLM-
generated, and not presented as human-verified ground truth.

This intentionally does NOT include a `ground_truth_explanation` field.
Writing 80-100 reference explanations by hand isn't feasible, and having
an LLM write them would make any embedding-similarity-to-reference metric
partly circular (grading a language model's explanation against another
language model's explanation of the same event). `evaluate_real.py`
instead scores real events on citation-verification-based metrics
(precision/coverage/unsupported-claim-rate) and retrieval metrics
(Recall@k/MRR@k against a same-ticker-same-event-type relevance proxy),
none of which require a fabricated reference text.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import get_settings
from core.exceptions import NewsRetrievalError
from processors.anomaly_detector import AnomalyDetector, ZScoreStrategy
from processors.data_loader import DataLoader, YFinanceDataStrategy
from processors.news_retriever import NewsRetrieverFactory

DEFAULT_TICKERS = [
    "NVDA", "TSLA", "AAPL", "META", "JPM",
    "XOM", "PFE", "AMZN", "NFLX", "BA",
]

# Deterministic, reproducible fact extraction -- no LLM involved. A
# fixed vocabulary of market-moving-action words plus regex for
# percentages/dollar amounts pulled straight out of the real headline.
ACTION_VOCAB = [
    "beats", "beat", "misses", "miss", "raises", "raised", "cuts", "cut",
    "downgrade", "downgraded", "upgrade", "upgraded", "recall", "recalls",
    "lawsuit", "investigation", "resigns", "resignation", "acquisition",
    "acquires", "merger", "guidance", "dividend", "buyback", "bankruptcy",
    "breach", "hack", "strike", "tariff", "earnings", "revenue", "profit",
    "loss", "layoffs", "restructuring", "delay", "delays", "approval",
    "rejected", "surge", "plunge", "soar", "slump", "warns", "warning",
]
PERCENT_RE = re.compile(r"\d+(?:\.\d+)?%")
DOLLAR_RE = re.compile(r"\$\d[\d,.]*\s?(?:billion|million|B|M|K)?", re.IGNORECASE)


def extract_key_facts(headline: str) -> List[str]:
    if not headline:
        return []
    facts = set(PERCENT_RE.findall(headline)) | set(DOLLAR_RE.findall(headline))
    lower = headline.lower()
    for word in ACTION_VOCAB:
        if re.search(rf"\b{re.escape(word)}\b", lower):
            facts.add(word)
    return sorted(facts)


def build_events_for_ticker(ticker: str, benchmark: str, start_date: str, end_date: str,
                             z_threshold: float, max_candidates: int, request_delay: float) -> List[Dict[str, Any]]:
    loader = DataLoader(YFinanceDataStrategy())
    data = loader.load_market_data(ticker, benchmark, start_date, end_date)

    detector = AnomalyDetector(ZScoreStrategy(z_threshold=z_threshold))
    events = detector.detect(data['processed_data'])
    if events.empty:
        return []

    settings = get_settings()
    if not settings.finnhub_api_key:
        raise RuntimeError("FINNHUB_API_KEY is required for real historical news lookup")
    news_retriever = NewsRetrieverFactory.create_retriever('finnhub', settings.finnhub_api_key)

    # Prioritize the largest anomalies first (most likely to have real news
    # coverage anyway) and cap how many get a news lookup, since Finnhub's
    # free tier is rate-limited (60 req/min) and this script hits it once
    # per candidate event across 10 tickers.
    candidates = events.assign(abs_z=events['Z_score'].abs()).sort_values('abs_z', ascending=False)
    candidates = candidates.head(max_candidates)

    records = []
    for _, row in candidates.iterrows():
        try:
            headline = news_retriever.retrieve_news(ticker, row['Date'])
        except NewsRetrievalError:
            headline = ''
        time.sleep(request_delay)

        if not headline.strip():
            continue

        record = row.to_dict()
        record['News_Headlines'] = headline
        record['Date'] = str(record['Date'])
        record['key_facts'] = extract_key_facts(headline)
        records.append(record)

    return records


def assemble_cases(ticker: str, records: List[Dict[str, Any]], max_per_ticker: int) -> List[Dict[str, Any]]:
    # Prefer events with more extracted facts (richer, more evaluable
    # headlines) when a ticker has more candidates than max_per_ticker.
    records = sorted(records, key=lambda r: len(r['key_facts']), reverse=True)[:max_per_ticker]

    cases = []
    for i, record in enumerate(records):
        history = [r for r in records if r is not record]
        cases.append({
            "id": f"real_{ticker}_{i}",
            "ticker": ticker,
            "event": {
                "Date": record["Date"],
                "Ticker": record["Ticker"],
                "Event_Type": record["Event_Type"],
                "Z_score": record["Z_score"],
                "Return": record.get("Return"),
                "News_Headlines": record["News_Headlines"],
            },
            "historical_context": [
                {
                    "Date": h["Date"], "Ticker": h["Ticker"], "Event_Type": h["Event_Type"],
                    "Z_score": h["Z_score"], "News_Headlines": h["News_Headlines"],
                }
                for h in history
            ],
            "key_facts": record["key_facts"],
        })
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Source a real-event evaluation set from live market/news data")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--benchmark", default="SPY")
    # Finnhub's free tier only has company-news data for roughly the last
    # year (empirically confirmed during development: real data at 11.5
    # months back, empty at 12 months back) -- not the multi-year archive
    # its date-range parameters might suggest. These defaults stay safely
    # inside that actual window; widen only if your Finnhub plan supports it.
    parser.add_argument("--start-date", default="2025-09-01")
    parser.add_argument("--end-date", default="2026-08-09")
    parser.add_argument("--z-threshold", type=float, default=2.5)
    parser.add_argument("--max-per-ticker", type=int, default=12)
    parser.add_argument("--max-candidates-per-ticker", type=int, default=20,
                         help="cap on how many of each ticker's anomalies get a news lookup (rate-limit control)")
    parser.add_argument("--request-delay", type=float, default=1.1,
                         help="seconds between Finnhub calls (free tier: 60 req/min)")
    parser.add_argument("--output", default=str(Path(__file__).parent / "real_eval_set.json"))
    args = parser.parse_args()

    all_cases = []
    for ticker in args.tickers:
        print(f"Sourcing events for {ticker}...", end=" ", flush=True)
        try:
            records = build_events_for_ticker(
                ticker, args.benchmark, args.start_date, args.end_date, args.z_threshold,
                args.max_candidates_per_ticker, args.request_delay,
            )
        except Exception as e:
            print(f"FAILED ({e})")
            continue

        cases = assemble_cases(ticker, records, args.max_per_ticker)
        all_cases.extend(cases)
        print(f"{len(records)} candidate events, kept {len(cases)}")

    output = {
        "note": (
            "REAL market events (yfinance) and REAL contemporaneous news headlines "
            "(Finnhub company-news, historical date-ranged), no fabricated data. "
            "'key_facts' are deterministically "
            "extracted from each event's own headline (regex + fixed action-word "
            "vocabulary in build_real_eval_set.py) -- not human-verified, not LLM-"
            "generated. There is no ground_truth_explanation field; see "
            "evaluate_real.py for the metrics this eval set is scored on and why."
        ),
        "source_config": {
            "tickers": args.tickers, "benchmark": args.benchmark,
            "start_date": args.start_date, "end_date": args.end_date,
            "z_threshold": args.z_threshold,
        },
        "cases": all_cases,
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nWrote {len(all_cases)} real events to {output_path}")


if __name__ == "__main__":
    main()
