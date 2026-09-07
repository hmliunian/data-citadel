def test_web_first_start_without_expert_config(tmp_path):
    from fastapi.testclient import TestClient

    from data_citadel.api import create_app
    from data_citadel.settings import Settings

    settings = Settings(dataset_root=tmp_path, experts_path=tmp_path / "missing.json",
                        artifacts_dir=tmp_path / "artifacts")
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/v1/inventory").json()["total_episodes"] == 0
        assert client.app.state.service.experts.version == "unconfigured"
