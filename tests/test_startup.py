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


def test_default_mode_stays_main_until_multiview_is_validated(monkeypatch):
    from data_citadel.settings import Settings

    monkeypatch.delenv("CITADEL_CAMERA_MODE", raising=False)
    assert Settings().camera_mode == "main"
    monkeypatch.setenv("CITADEL_CAMERA_MODE", "main_wrist")
    assert Settings().camera_mode == "main_wrist"
