import importlib
import sys

import pytest


class StubBinaryClassifier:
    """Avoid loading a trained model while exercising the Flask route."""

    def classify(self, text):
        return "positive", 0.9


@pytest.fixture
def api_module(monkeypatch):
    import src.models

    monkeypatch.setattr(src.models, "BinaryClassifier", StubBinaryClassifier)
    sys.modules.pop("src.app", None)
    module = importlib.import_module("src.app")
    module.app.config.update(TESTING=True)
    yield module
    sys.modules.pop("src.app", None)


@pytest.fixture
def client(api_module):
    return api_module.app.test_client()


@pytest.mark.parametrize("body", ["null", "[]", '"sentiment"'])
def test_analyze_rejects_json_values_other_than_an_object(client, body):
    response = client.post("/api/analyze", data=body, content_type="application/json")

    assert response.status_code == 400
    assert response.is_json
    assert response.get_json() == {"error": "JSON body must be an object"}


def test_analyze_rejects_a_non_json_content_type(client):
    response = client.post(
        "/api/analyze",
        data='{"text": "A good day"}',
        content_type="text/plain",
    )

    assert response.status_code == 415
    assert response.is_json
    assert response.get_json() == {"error": "Content-Type must be application/json"}


def test_analyze_returns_json_for_malformed_json(client):
    response = client.post("/api/analyze", data="{", content_type="application/json")

    assert response.status_code == 400
    assert response.is_json
    assert response.get_json() == {"error": "JSON body must be an object"}


def test_analyze_returns_json_for_an_empty_json_body(client):
    response = client.post("/api/analyze", data="", content_type="application/json")

    assert response.status_code == 400
    assert response.is_json
    assert response.get_json() == {"error": "JSON body must be an object"}


@pytest.mark.parametrize("payload", [{}, {"text": ""}])
def test_analyze_rejects_missing_or_empty_text(client, payload):
    response = client.post("/api/analyze", json=payload)

    assert response.status_code == 400
    assert response.is_json
    assert response.get_json() == {"error": "No text provided"}


@pytest.mark.parametrize("text", [None, 123, False, ["not", "text"]])
def test_analyze_rejects_non_string_text(client, text):
    response = client.post("/api/analyze", json={"text": text})

    assert response.status_code == 400
    assert response.is_json
    assert response.get_json() == {"error": "Text must be a string"}


def test_analyze_rejects_whitespace_only_text(client):
    response = client.post("/api/analyze", json={"text": " \t\n "})

    assert response.status_code == 400
    assert response.is_json
    assert response.get_json() == {"error": "No text provided"}


def test_analyze_rejects_text_longer_than_ten_thousand_characters(client):
    response = client.post("/api/analyze", json={"text": "a" * 10_001})

    assert response.status_code == 413
    assert response.is_json
    assert response.get_json() == {
        "error": "Text exceeds maximum length of 10000 characters"
    }


def test_analyze_accepts_text_at_the_maximum_length(client):
    response = client.post("/api/analyze", json={"text": "a" * 10_000})

    assert response.status_code == 200


def test_analyze_returns_results_for_valid_text(client):
    response = client.post("/api/analyze", json={"text": "A good day"})

    assert response.status_code == 200
    assert response.get_json() == {
        "sentiment": "Positive",
        "sentiment_confidence": 0.9,
        "emotion": "System Not Ready",
        "emotion_confidence": 0.0,
    }


def test_health_and_analysis_preserve_unavailable_model_fallback(api_module, client):
    api_module.binary_clf = None
    api_module.emotion_model = None
    api_module.emotion_vectorizer = None

    health = client.get("/health")
    analysis = client.post("/api/analyze", json={"text": "A good day"})

    assert health.get_json() == {
        "status": "healthy",
        "binary_loaded": False,
        "emotion_loaded": False,
    }
    assert analysis.status_code == 200
    assert analysis.get_json() == {
        "sentiment": "Error",
        "sentiment_confidence": 0.0,
        "emotion": "System Not Ready",
        "emotion_confidence": 0.0,
    }
