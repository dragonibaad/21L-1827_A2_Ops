import os
os.environ.setdefault("VISIONGUARD_DATABASE_URL", "sqlite:///:memory:")

from app import app  # noqa: E402  (import comes after the env var on purpose)


def test_health_route():
    client = app.test_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"