import importlib.util
import pathlib
import sys

import pytest

training = pathlib.Path(__file__).parents[1] / "training"
sys.path.insert(0, str(training))
pytest.importorskip("mujoco")
pytest.importorskip("stable_baselines3")
spec = importlib.util.spec_from_file_location("eval_robustness", training / "eval_robustness.py")
er = importlib.util.module_from_spec(spec)
spec.loader.exec_module(er)


def test_wilson_bounds():
    lo, hi = er.wilson(0, 16)
    assert lo == 0.0 and 0.1 < hi < 0.25
    lo, hi = er.wilson(16, 16)
    assert hi == 1.0 and lo > 0.75
    lo, hi = er.wilson(8, 16)
    assert lo < 0.5 < hi


def test_wilson_empty():
    assert er.wilson(0, 0) == (0.0, 1.0)
