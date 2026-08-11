"""
Airflow DAG orchestrating the anomaly -> retrieval -> grounded-explanation
pipeline as 5 tasks, each a thin PythonOperator calling the *same*
existing classes (DataLoader, AnomalyDetector, NewsRetrieverFactory,
RAGRetriever, GroundedGroqExplanationStrategy, BaseDataRepository) the
API/CLI/Streamlit consumers use -- a real scheduled-orchestration
consumer of the same Strategy interfaces, not a parallel pipeline.

Intermediate state is handed between tasks by filename via XCom (each
task saves to results/, the next loads by the pushed filename) rather
than passing DataFrames through XCom directly, which Airflow does not
handle well at any real size.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator

from config.settings import get_settings
from core.base import BaseDataRepository
from processors.anomaly_detector import AnomalyDetector, ZScoreStrategy
from processors.chunking import ChunkerFactory
from processors.data_loader import DataLoader, YFinanceDataStrategy
from processors.embedding_generator import EmbeddingGeneratorFactory
from processors.grounded_explainer import GroundedGroqExplanationStrategy
from processors.news_retriever import NewsRetrieverFactory
from processors.rag_retriever import RAGRetriever
from processors.reranker import RerankerFactory
from processors.vector_store import VectorStoreFactory

REPOSITORY = BaseDataRepository()

default_args = {
    "owner": "financial-anomaly-detection-rag",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _ticker() -> str:
    return Variable.get("faildrag_ticker", default_var="TSLA")


def _benchmark() -> str:
    return Variable.get("faildrag_benchmark", default_var="SPY")


def load_data(ti, **context) -> None:
    settings = get_settings()
    ticker, benchmark = _ticker(), _benchmark()
    start_date = Variable.get("faildrag_start_date", default_var=settings.default_start_date)
    end_date = Variable.get("faildrag_end_date", default_var=settings.default_end_date)

    loader = DataLoader(YFinanceDataStrategy())
    data = loader.load_market_data(ticker, benchmark, start_date, end_date)

    filename = f"airflow_processed_{context['ds_nodash']}.parquet"
    REPOSITORY.save_events(data['processed_data'], filename)
    ti.xcom_push(key="processed_data_file", value=filename)


def detect_anomalies(ti, **context) -> None:
    processed_file = ti.xcom_pull(key="processed_data_file", task_ids="load_data")
    processed_data = REPOSITORY.load_events(processed_file)

    detector = AnomalyDetector(ZScoreStrategy(z_threshold=2.5))
    events = detector.detect(processed_data)

    filename = f"airflow_events_{context['ds_nodash']}.parquet"
    REPOSITORY.save_events(events, filename)
    ti.xcom_push(key="events_file", value=filename)
    ti.xcom_push(key="num_events", value=len(events))


def retrieve_and_chunk(ti, **context) -> None:
    """News retrieval -- the text that gets chunked by RAGRetriever in the
    next task (chunking itself needs the embedding/vector-store components
    together, so it's done there rather than duplicating that setup here)."""
    events_file = ti.xcom_pull(key="events_file", task_ids="detect_anomalies")
    events = REPOSITORY.load_events(events_file)

    if events.empty:
        filename = events_file
    else:
        settings = get_settings()
        if settings.finnhub_api_key:
            news_retriever = NewsRetrieverFactory.create_retriever('finnhub', settings.finnhub_api_key)
        else:
            news_retriever = NewsRetrieverFactory.create_retriever('yahoo')
        events_with_news = news_retriever.add_news_to_events(events, _ticker())
        filename = f"airflow_events_with_news_{context['ds_nodash']}.parquet"
        REPOSITORY.save_events(events_with_news, filename)

    ti.xcom_push(key="events_with_news_file", value=filename)


def explain_with_citations(ti, **context) -> None:
    events_file = ti.xcom_pull(key="events_with_news_file", task_ids="retrieve_and_chunk")
    events_with_news = REPOSITORY.load_events(events_file)

    settings = get_settings()
    explanations: dict = {}

    if not events_with_news.empty and settings.groq_api_key:
        chunker = ChunkerFactory.create_chunker(settings.chunking_strategy)
        embedding_generator = EmbeddingGeneratorFactory.create_generator('sentence_transformer')
        # numpy backend: no PersistentClient state to share across the
        # separate process each Airflow task runs in.
        vector_store = VectorStoreFactory.create('numpy')
        reranker = RerankerFactory.create_reranker('none')  # keep the scheduled run dependency-light
        retriever = RAGRetriever(chunker, embedding_generator, vector_store, reranker)
        strategy = GroundedGroqExplanationStrategy(settings.groq_api_key, retriever)

        recent_indices = list(range(max(0, len(events_with_news) - 3), len(events_with_news)))
        for position in recent_indices:
            event_data = events_with_news.iloc[position].to_dict()
            history = events_with_news.drop(events_with_news.index[position]).to_dict('records')
            explanations[str(position)] = strategy.explain_event(event_data, history)

    filename = f"airflow_explanations_{context['ds_nodash']}.json"
    with open(f"{REPOSITORY.base_path}/{filename}", 'w') as f:
        json.dump(explanations, f, indent=2)
    ti.xcom_push(key="explanations_file", value=filename)


def save_results(ti, **context) -> None:
    num_events = ti.xcom_pull(key="num_events", task_ids="detect_anomalies") or 0
    explanations_file = ti.xcom_pull(key="explanations_file", task_ids="explain_with_citations")

    summary = {
        "run_date": context["ds"],
        "ticker": _ticker(),
        "benchmark": _benchmark(),
        "num_events_detected": num_events,
        "explanations_file": explanations_file,
    }
    filename = f"airflow_run_summary_{context['ds_nodash']}.json"
    with open(f"{REPOSITORY.base_path}/{filename}", 'w') as f:
        json.dump(summary, f, indent=2)


with DAG(
    dag_id="financial_anomaly_dag",
    description="Detect anomalies, retrieve context, and generate grounded explanations",
    default_args=default_args,
    schedule="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["financial-anomaly-detection-rag"],
) as dag:
    t1 = PythonOperator(task_id="load_data", python_callable=load_data)
    t2 = PythonOperator(task_id="detect_anomalies", python_callable=detect_anomalies)
    t3 = PythonOperator(task_id="retrieve_and_chunk", python_callable=retrieve_and_chunk)
    t4 = PythonOperator(task_id="explain_with_citations", python_callable=explain_with_citations)
    t5 = PythonOperator(task_id="save_results", python_callable=save_results)

    t1 >> t2 >> t3 >> t4 >> t5
