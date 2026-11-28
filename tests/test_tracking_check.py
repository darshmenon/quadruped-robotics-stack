import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
import tracking_check as tc  # noqa: E402


def row(vx=0.2, falls=0, samples=100):
    return dict(episodes=4, falls=falls, samples=samples, vx_mean=vx)


def test_verdict_flags_standing_and_falling():
    assert tc.verdict(row(vx=0.007), 0.2) == "stands"
    assert tc.verdict(row(vx=-0.2), 0.2) == "stands"
    assert tc.verdict(row(vx=0.243), 0.2) == "ok"
    assert tc.verdict(row(falls=2), 0.2) == "falls"
    assert tc.verdict(dict(episodes=4, falls=4, samples=0), 0.35) == "falls"
    assert tc.verdict(row(vx=0.008), 0.1) == "not_judged"


def test_run_config_saved_per_dir_and_appended(tmp_path):
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir()
    tc.save_run_config([str(tmp_path), str(ckpt)], {"legs_only": False}, {"GO2_TROT_PRIOR": "1", "HOME": "x"})
    tc.save_run_config([str(tmp_path), str(ckpt)], {"legs_only": True}, {"GO2_PUSH_VEL": "0.3"})
    for model in (tmp_path / "best_model.zip", ckpt / "go2_mujoco_50000_steps.zip"):
        entry = tc.find_run_config(str(model))
        assert entry["options"] == {"legs_only": True}
        assert entry["env"] == {"GO2_PUSH_VEL": "0.3"}
    assert tc.env_mismatches(entry, {"GO2_PUSH_VEL": "0.3"}) == []
    assert tc.env_mismatches(entry, {"GO2_TROT_PRIOR": "1"}) == [
        ("GO2_PUSH_VEL", "0.3", None), ("GO2_TROT_PRIOR", None, "1")]


def test_no_run_config(tmp_path):
    assert tc.find_run_config(str(tmp_path / "m.zip")) is None
