import pandas as pd

from processors.anomaly_detector import AnomalyDetector, IsolationForestStrategy, ZScoreStrategy


def make_data():
    dates = pd.date_range("2025-01-01", periods=6)
    return pd.DataFrame({
        "Ticker": ["TEST"] * 6,
        "Return": [0.001, -0.002, 0.15, 0.0, -0.001, 0.30],
        "Z_score": [0.1, -0.2, 3.0, 0.0, -0.1, 4.2],
        "Rolling_STD": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        "Close": [100, 100.1, 105, 105, 104.9, 110],
    }, index=dates)


class TestZScoreStrategy:
    def test_flags_positive_and_negative_outliers(self):
        strategy = ZScoreStrategy(z_threshold=2.5)
        events = strategy.detect(make_data())

        event_types = events['Event_Type'].tolist()
        assert 'Positive Outlier' in event_types
        assert event_types.count('Positive Outlier') == 2  # rows with Z=3.0 and Z=4.2

    def test_flags_volatility_spike_below_z_threshold(self):
        data = make_data()
        data.loc[data.index[1], 'Return'] = 0.05  # 5x rolling std, but Z-score stays low
        data.loc[data.index[1], 'Z_score'] = 0.3
        strategy = ZScoreStrategy(z_threshold=2.5)

        events = strategy.detect(data)

        matched = events[events['Date'] == data.index[1]]
        assert not matched.empty
        assert matched.iloc[0]['Event_Type'] == 'Volatility Spike'

    def test_no_events_when_nothing_crosses_threshold(self):
        data = make_data()
        data['Z_score'] = 0.1
        data['Return'] = 0.001
        strategy = ZScoreStrategy(z_threshold=2.5)

        events = strategy.detect(data)

        assert events.empty

    def test_get_and_set_parameters(self):
        strategy = ZScoreStrategy(z_threshold=2.5)
        strategy.set_parameters(z_threshold=3.0, vol_window=20, vol_multiplier=1.5)
        params = strategy.get_parameters()
        assert params == {'z_threshold': 3.0, 'vol_window': 20, 'vol_multiplier': 1.5}


class TestAnomalyDetector:
    def test_detect_delegates_to_strategy(self):
        detector = AnomalyDetector(ZScoreStrategy(z_threshold=2.5))
        events = detector.detect(make_data())
        assert not events.empty

    def test_analyze_events_counts_by_type(self):
        detector = AnomalyDetector(ZScoreStrategy(z_threshold=2.5))
        events = detector.detect(make_data())
        summary = detector.analyze_events(events)
        assert summary['total_events'] == len(events)
        assert summary['positive_outliers'] + summary['negative_outliers'] + summary['volatility_spikes'] == len(events)

    def test_analyze_events_handles_empty(self):
        detector = AnomalyDetector(ZScoreStrategy())
        summary = detector.analyze_events(pd.DataFrame())
        assert summary['total_events'] == 0

    def test_set_parameters_delegates_to_strategy(self):
        strategy = ZScoreStrategy()
        detector = AnomalyDetector(strategy)
        detector.set_parameters(z_threshold=1.5)
        assert strategy.z_threshold == 1.5

    def test_isolation_forest_strategy_runs_without_error(self):
        detector = AnomalyDetector(IsolationForestStrategy(contamination=0.3))
        data = make_data()
        data['Benchmark_Return'] = 0.0
        events = detector.detect(data)
        assert isinstance(events, pd.DataFrame)
