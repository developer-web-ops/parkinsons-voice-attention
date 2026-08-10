from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["n_features"] == 753


def test_features_and_examples():
    features = client.get("/api/features").json()
    assert "Baseline Features" in features["groups"]
    assert set(features["defaults"]) == set(features["min"]) == set(features["max"])

    examples = client.get("/api/examples").json()
    assert {"healthy", "parkinsons"} <= set(examples)


def test_predict_defaults_and_attention():
    body = client.post("/api/predict", json={"features": {}, "model": "attention"}).json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["prediction"] in (0, 1)
    assert abs(sum(body["attention"].values()) - 1.0) < 1e-4
    assert len(body["shap"]) > 0


def test_predict_examples_are_classified_correctly():
    examples = client.get("/api/examples").json()
    healthy = client.post(
        "/api/predict", json={"features": examples["healthy"], "explain": False}
    ).json()
    pd_case = client.post(
        "/api/predict", json={"features": examples["parkinsons"], "explain": False}
    ).json()
    assert pd_case["probability"] > healthy["probability"]


def test_predict_mlp_has_no_attention():
    body = client.post("/api/predict", json={"features": {}, "model": "mlp"}).json()
    assert body["attention"] is None


def test_predict_rejects_bad_input():
    assert client.post("/api/predict", json={"features": {}, "model": "nope"}).status_code == 400
    assert (
        client.post("/api/predict", json={"features": {"not_a_feature": 1.0}}).status_code == 400
    )


def test_predict_csv():
    import csv
    import io

    features = client.get("/api/features").json()["defaults"]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(features))
    writer.writeheader()
    writer.writerow(features)
    writer.writerow(features)

    response = client.post(
        "/api/predict/csv",
        files={"file": ("sample.csv", buffer.getvalue(), "text/csv")},
    )
    body = response.json()
    assert body["count"] == 2
    assert body["positive"] + body["negative"] == 2


def test_predict_csv_rejects_unrecognised_header():
    response = client.post(
        "/api/predict/csv", files={"file": ("bad.csv", "a,b\n1,2\n", "text/csv")}
    )
    assert response.status_code == 400


def test_metrics_and_reports():
    metrics = client.get("/api/metrics").json()
    assert {"mlp", "attention", "logistic_regression"} <= set(metrics)
    assert 0.0 <= metrics["attention"]["roc_auc"] <= 1.0
    assert client.get("/api/report/roc_curves.png").status_code == 200
    assert client.get("/api/report/../requirements.txt").status_code == 404


def test_frontend_served():
    assert "Voice Biomarkers" in client.get("/").text
