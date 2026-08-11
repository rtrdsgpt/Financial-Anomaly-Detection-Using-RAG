import json

import pytest

from experiments.evaluate_baselines import (
    CONFIG_NAMES,
    compute_summary,
    load_existing_results,
    replace_case_result,
    successful_case_ids,
    with_retries,
    write_report,
)


class TestLoadExistingResults:
    def test_missing_file_returns_empty_structure(self, tmp_path):
        result = load_existing_results(tmp_path / "does_not_exist.json")
        assert result == {name: [] for name in CONFIG_NAMES}

    def test_loads_existing_report(self, tmp_path):
        path = tmp_path / "report.json"
        path.write_text(json.dumps({"results": {"llm_only": [{"case_id": "a", "fact_overlap": 0.5}]}}))

        result = load_existing_results(path)

        assert result["llm_only"] == [{"case_id": "a", "fact_overlap": 0.5}]
        assert result["rag_with_reranker"] == []


class TestSuccessfulCaseIds:
    def test_case_must_succeed_in_every_config(self):
        all_results = {
            "llm_only": [{"case_id": "a", "fact_overlap": 1.0}, {"case_id": "b", "fact_overlap": 1.0}],
            "llm_legacy_retrieval": [{"case_id": "a", "fact_overlap": 1.0}, {"case_id": "b", "fact_overlap": 1.0}],
            "rag_no_reranker": [{"case_id": "a", "fact_overlap": 1.0}, {"case_id": "b", "error": "boom"}],
            "rag_with_reranker": [{"case_id": "a", "fact_overlap": 1.0}, {"case_id": "b", "fact_overlap": 1.0}],
        }

        assert successful_case_ids(all_results) == {"a"}

    def test_empty_results_yields_no_successes(self):
        all_results = {name: [] for name in CONFIG_NAMES}
        assert successful_case_ids(all_results) == set()


class TestReplaceCaseResult:
    def test_replaces_existing_entry_for_same_case(self):
        all_results = {"llm_only": [{"case_id": "a", "error": "boom"}]}
        replace_case_result(all_results, "llm_only", "a", {"case_id": "a", "fact_overlap": 0.9})
        assert all_results["llm_only"] == [{"case_id": "a", "fact_overlap": 0.9}]

    def test_appends_when_case_not_present(self):
        all_results = {"llm_only": [{"case_id": "a", "fact_overlap": 0.5}]}
        replace_case_result(all_results, "llm_only", "b", {"case_id": "b", "fact_overlap": 0.1})
        assert [r["case_id"] for r in all_results["llm_only"]] == ["a", "b"]


class TestComputeSummary:
    def test_excludes_errors_from_means_but_counts_them(self):
        all_results = {
            "llm_only": [{"case_id": "a", "fact_overlap": 1.0}, {"case_id": "b", "error": "boom"}],
        }
        summary = compute_summary(all_results)
        assert summary["llm_only"]["num_cases"] == 2
        assert summary["llm_only"]["num_successful"] == 1
        assert summary["llm_only"]["mean_fact_overlap"] == 1.0

    def test_includes_citation_metrics_only_when_present(self):
        all_results = {
            "rag_no_reranker": [{"case_id": "a", "fact_overlap": 1.0, "citation_precision": 1.0,
                                   "citation_coverage": 0.5, "unsupported_claim_rate": 0.5}],
            "llm_only": [{"case_id": "a", "fact_overlap": 1.0}],
        }
        summary = compute_summary(all_results)
        assert "mean_citation_precision" in summary["rag_no_reranker"]
        assert "mean_citation_precision" not in summary["llm_only"]


class TestWriteReport:
    def test_writes_and_returns_summary(self, tmp_path):
        output_path = tmp_path / "nested" / "report.json"
        all_results = {"llm_only": [{"case_id": "a", "fact_overlap": 1.0}]}

        summary = write_report(output_path, all_results)

        assert output_path.exists()
        on_disk = json.loads(output_path.read_text())
        assert on_disk["summary"] == summary
        assert on_disk["results"] == all_results


class TestWithRetriesFailsFastOnDailyQuota:
    def test_does_not_retry_tpd_error(self):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            raise RuntimeError("Rate limit reached ... on tokens per day (TPD): Limit 200000")

        with pytest.raises(RuntimeError):
            with_retries(flaky, attempts=3, base_delay=0.01)

        assert calls["n"] == 1

    def test_retries_transient_error(self):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 2:
                raise RuntimeError("temporary network error")
            return "ok"

        result = with_retries(flaky, attempts=3, base_delay=0.01)

        assert result == "ok"
        assert calls["n"] == 2
