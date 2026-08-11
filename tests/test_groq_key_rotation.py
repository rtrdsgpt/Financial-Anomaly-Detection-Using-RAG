import pytest

from processors.groq_key_rotation import RotatingGroqClient, is_rate_limit_error, parse_api_keys


class TestParseApiKeys:
    def test_splits_comma_separated_keys(self):
        assert parse_api_keys("gsk_abc,gsk_def") == ["gsk_abc", "gsk_def"]

    def test_single_key_no_comma(self):
        assert parse_api_keys("gsk_abc") == ["gsk_abc"]

    def test_strips_whitespace(self):
        assert parse_api_keys("gsk_abc, gsk_def ,  gsk_ghi") == ["gsk_abc", "gsk_def", "gsk_ghi"]

    def test_empty_or_none_returns_empty_list(self):
        assert parse_api_keys("") == []
        assert parse_api_keys(None) == []


class TestIsRateLimitError:
    @pytest.mark.parametrize("message", [
        "Error code: 429 - rate_limit_exceeded",
        "Rate limit reached for requests per day (RPD)",
        "You have exceeded your daily quota",
        "Tokens per day (TPD) limit exceeded",
    ])
    def test_detects_rate_limit_messages(self, message):
        assert is_rate_limit_error(message) is True

    @pytest.mark.parametrize("message", [
        "Invalid API Key",
        "model_not_found",
        "Failed to validate JSON",
    ])
    def test_does_not_flag_other_errors(self, message):
        assert is_rate_limit_error(message) is False


class _FakeCompletions:
    def __init__(self, behavior):
        self._behavior = behavior  # list of callables, one per call
        self.calls = 0

    def create(self, **kwargs):
        action = self._behavior[min(self.calls, len(self._behavior) - 1)]
        self.calls += 1
        return action()


class _FakeGroq:
    """Stands in for groq.Groq -- one instance per api_key, with
    per-key call behavior driven by a shared script keyed by api_key.
    Mirrors the real client.chat.completions.create(...) shape."""

    instances = {}

    def __init__(self, api_key):
        self.api_key = api_key
        completions = _FakeCompletions(_FakeGroq.instances[api_key])
        self.chat = type("Chat", (), {"completions": completions})()


class TestRotatingGroqClient:
    def test_single_key_success_no_rotation(self, monkeypatch):
        _FakeGroq.instances = {"key1": [lambda: "ok"]}
        monkeypatch.setattr("groq.Groq", _FakeGroq)

        client = RotatingGroqClient(["key1"])
        result = client.create_chat_completion(model="m", messages=[])

        assert result == "ok"
        assert client._index == 0

    def test_rotates_to_next_key_on_rate_limit(self, monkeypatch):
        def raise_rate_limit():
            raise RuntimeError("Error code: 429 - rate_limit_exceeded")

        _FakeGroq.instances = {
            "key1": [raise_rate_limit],
            "key2": [lambda: "success-on-key2"],
        }
        monkeypatch.setattr("groq.Groq", _FakeGroq)

        client = RotatingGroqClient(["key1", "key2"])
        result = client.create_chat_completion(model="m", messages=[])

        assert result == "success-on-key2"
        assert client._index == 1

    def test_raises_when_all_keys_rate_limited(self, monkeypatch):
        def raise_rate_limit():
            raise RuntimeError("rate limit exceeded")

        _FakeGroq.instances = {"key1": [raise_rate_limit], "key2": [raise_rate_limit]}
        monkeypatch.setattr("groq.Groq", _FakeGroq)

        client = RotatingGroqClient(["key1", "key2"])
        with pytest.raises(RuntimeError, match="rate limit"):
            client.create_chat_completion(model="m", messages=[])

    def test_does_not_rotate_on_non_rate_limit_error(self, monkeypatch):
        def raise_auth_error():
            raise RuntimeError("Invalid API Key")

        _FakeGroq.instances = {
            "key1": [raise_auth_error],
            "key2": [lambda: "should not be reached"],
        }
        monkeypatch.setattr("groq.Groq", _FakeGroq)

        client = RotatingGroqClient(["key1", "key2"])
        with pytest.raises(RuntimeError, match="Invalid API Key"):
            client.create_chat_completion(model="m", messages=[])
        assert client._index == 0

    def test_requires_at_least_one_key(self):
        with pytest.raises(ValueError):
            RotatingGroqClient([])
