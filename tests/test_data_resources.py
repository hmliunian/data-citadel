import httpx
import pytest

from citadel.infrastructure.files import read
from citadel.infrastructure.datasets import inventory, load_manifest, prepare
from citadel.domain.tasks import profile_for
from citadel.infrastructure.resources import fetch

PROFILES = {"grasp": {"action_ids": ["A_001"], "success": "悬空", "allowed": "换手",
                      "failures": ["失败"], "hold_seconds": 2, "hold_tolerance_s": 0.2}}


def test_manifest_is_disjoint_reproducible_and_tamper_evident(dataset, tmp_path):
    first = prepare(dataset, tmp_path / "one")
    second = prepare(dataset, tmp_path / "two")
    assert first["splits"] == second["splits"]
    a, b = map(set, first["splits"].values())
    assert a.isdisjoint(b) and len(a | b) == 8
    assert load_manifest(tmp_path / "one") == first
    file = tmp_path / "one/manifest.json"
    file.write_text(file.read_text().replace('"interval_s": 1.0', '"interval_s": 0.1'))
    with pytest.raises(ValueError, match="changed"):
        load_manifest(tmp_path / "one")


def test_source_path_cannot_escape_episode(dataset):
    file = next((dataset / "receipts").glob("*.json"))
    file.write_text(file.read_text().replace("episode.mcap", "../../../../private.mcap"))
    with pytest.raises(ValueError, match="path"):
        inventory(dataset)


def test_incomplete_download_blocks_inventory(dataset):
    file = next((dataset / "receipts").glob("*.json"))
    file.write_text(file.read_text().replace('"complete": true', '"complete": false'))
    with pytest.raises(ValueError, match="receipt"):
        inventory(dataset)


def test_task_extension_is_configuration_driven():
    task = {"steps": [{"action_id": "A_009"}]}
    with pytest.raises(ValueError, match="configured"):
        profile_for(task, PROFILES)
    profiles = PROFILES | {"wipe": {"action_ids": ["A_009"], "success": "擦拭完成",
                                    "allowed": "不同擦拭路径", "failures": ["没有接触"]}}
    assert profile_for(task, profiles)["name"] == "wipe"


def test_resources_use_actual_task_and_do_not_persist_signed_urls(tmp_path, jpeg):
    requests = []
    def respond(request):
        requests.append(str(request.url))
        if request.url.path.endswith("/resources"):
            return httpx.Response(200, json={"code": 200, "data": {
                "task_code": "DL-TEST", "steps": [{"action_id": "A_001", "action_text": "拿起物体"}],
                "images": [{"id": str(i), "name": name, "type": name,
                            "url": f"https://images.test/{i}?signature=TEMP_SECRET"}
                           for i, name in enumerate(("object", "scene", "execution_location"))]}})
        return httpx.Response(200, content=jpeg)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        saved = fetch("DL-TEST", tmp_path, "https://tasks.test", client=client)
        assert len(saved["images"]) == 3
        assert fetch("DL-TEST", tmp_path, client=client) == saved
    assert len(requests) == 4
    assert requests[0].endswith("/code/DL-TEST/resources")
    assert "TEMP_SECRET" not in (tmp_path / "resources/DL-TEST/task.json").read_text()
    assert profile_for(saved, PROFILES)["name"] == "grasp"
    path = tmp_path / saved["images"][0]["path"]
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        fetch("DL-TEST", tmp_path)


@pytest.mark.parametrize("wrong_task", [True, False])
def test_wrong_or_missing_task_references_block_review(tmp_path, wrong_task):
    payload = {"code": 200, "data": {"task_code": "DL-OTHER" if wrong_task else "DL-TEST",
               "steps": [{"action_id": "A_001", "action_text": "拿起"}],
               "images": [{"type": "object"}]}}
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))) as client:
        with pytest.raises(ValueError):
            fetch("DL-TEST", tmp_path, "https://tasks.test", client=client)


def test_outputs_cannot_be_written_into_dataset(dataset):
    with pytest.raises(ValueError, match="read-only"):
        prepare(dataset, dataset / "artifacts")


def test_source_gt_is_retained_without_expert_requirement(dataset, tmp_path):
    manifest = prepare(dataset, tmp_path / "run")
    assert len(manifest["episodes"]) == 8
    assert {r["gt"] for r in manifest["episodes"].values()} == {"correct", "incorrect"}
    assert read(tmp_path / "run/manifest.json")["sampling"]["interval_s"] == 1


def test_task_local_action_ids_cannot_select_another_task():
    from citadel.domain.tasks import profile_for
    common = {"action_ids": ["A_001"], "allowed": "自由路径", "failures": ["未完成"]}
    profiles = {
        "grasp": {**common, "task_codes": ["DL-GRASP"], "success": "抓起"},
        "wipe": {**common, "task_codes": ["DL-WIPE"], "success": "擦拭"},
    }
    refs = {"task_code": "DL-WIPE", "steps": [{"action_id": "A_001"}]}
    assert profile_for(refs, profiles)["name"] == "wipe"
    refs["task_code"] = "DL-UNKNOWN"
    with pytest.raises(ValueError, match="exactly one"):
        profile_for(refs, profiles)
