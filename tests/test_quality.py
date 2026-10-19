import copy
import hashlib
import json

import pytest

from rocket_landing.quality import DEFAULT_POLICY, assess, audit


def cases():
    return [dict(scenario=scenario, episodes=50, landed_ci95=[0.8, 0.99],
                 timeout_rate=0.02, mean_base_return=230)
            for scenario in ("calm", "breezy", "gusty")]


def test_complete_reliable_model_passes():
    assert assess(cases(), DEFAULT_POLICY)["passed"]


@pytest.mark.parametrize("change", ["missing", "duplicate", "small", "uncertain", "timeout", "score", "nan"])
def test_rejects_weak_or_incomplete_evidence(change):
    rows = cases()
    if change == "missing": rows.pop()
    if change == "duplicate": rows.append(copy.deepcopy(rows[0]))
    if change == "small": rows[0]["episodes"] = 20
    if change == "uncertain": rows[2]["landed_ci95"][0] = 0.49
    if change == "timeout": rows[1]["timeout_rate"] = 0.21
    if change == "score": rows[0]["mean_base_return"] = 199
    if change == "nan": rows[0]["mean_base_return"] = float("nan")
    assert not assess(rows, DEFAULT_POLICY)["passed"]


def test_modified_weights_fail_and_incomplete_suite_rejected(tmp_path):
    model = tmp_path / "model.zip"
    model.write_bytes(b"original")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    rows = [dict(**row, model="Example", model_path=str(model), model_sha256=digest) for row in cases()]
    manifest = dict(status="complete", completed_cases=3, models=[str(model)], first_episode_seed=300000)
    (tmp_path / "results.json").write_text(json.dumps(rows))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert audit(tmp_path)["passed"]
    model.write_bytes(b"changed")
    assert not audit(tmp_path)["passed"]
    manifest["status"] = "running"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        audit(tmp_path)
