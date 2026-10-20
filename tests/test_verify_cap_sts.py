"""CPU checks for DSV41_VERIFY_CAP_STS (adapter/verify_cap.py's confidence calibration).

  PYTHONPATH=adapter python3 tests/test_verify_cap_sts.py

1. sigmoid(logit(p) / T) with the fitted T moves the cap's live length the way the fitter says
   (underconfident positions verify longer, overconfident shorter), and `conf:T` is applied to
   the calibrated values, not the raw ones;
2. T = 1 is the identity on the rule (same live lengths, same threshold decision) and the raw
   confidences still reach DSV41_VERIFY_CAP_LOG (`_state["last_conf"]`), so a refit sees the
   uncalibrated stream;
3. without DSV41_VERIFY_CAP_STS nothing changes -- `calibrate` is a pass-through;
4. the loader refuses non-positive temperatures and a temperature count that is neither 1 nor
   one per draft.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["DSV41_VERIFY_CAP"] = "conf:0.1"
import torch  # noqa: E402

import verify_cap as vc  # noqa: E402

GAMMA = vc.STRIDE - 1


def write_json(path, temperatures):
    path.write_text(json.dumps({"temperatures": temperatures}))
    return str(path)


def rule(conf, threshold=0.1, kmin=1):
    """The cap's own rule, for reference (verify_cap.set_live_from_confidence minus the buffer)."""
    cum = torch.cumprod(conf.float().clamp(0, 1), dim=1)
    return (cum >= threshold).to(torch.int64).cumprod(dim=1).sum(dim=1).clamp(min=kmin) + 1


def main():
    vc.share_live = lambda buf, bs: None          # no distributed group on CPU
    work = Path(tempfile.mkdtemp(prefix="vcap-sts-"))

    # A head that is overconfident early and underconfident late (the shape that makes a fixed
    # threshold cut the wrong drafts): calibration with the matching T must move the cut.
    conf = torch.tensor([[0.9, 0.9, 0.9, 0.2, 0.2],
                         [0.4, 0.4, 0.4, 0.4, 0.4],
                         [0.99, 0.3, 0.3, 0.3, 0.3]])

    # 3. without STS the module is a pass-through
    assert vc.calibrate(conf) is conf
    vc.set_live_from_confidence(conf, 3)
    raw_live = vc.live_buf()[:3].clone()

    # 1. with STS the rule reads the calibrated values
    temperatures = [0.5, 0.5, 0.5, 2.0, 2.0]
    vc._state["sts_t"] = torch.tensor(temperatures)
    calibrated = vc.calibrate(conf)
    expected = rule(calibrated)
    vc.set_live_from_confidence(conf, 3)
    assert torch.equal(vc.live_buf()[:3], expected), (vc.live_buf()[:3], expected)
    assert not torch.equal(vc.live_buf()[:3], raw_live), "calibration changed nothing"
    # temperature scaling moves every probability toward (T > 1) or away from (T < 1) 0.5,
    # which is the whole mechanism: the same threshold now cuts at a different place
    assert calibrated[0, 3] > conf[0, 3]           # 0.2 with T=2 -> 0.333, toward 0.5
    assert calibrated[0, 0] > conf[0, 0]           # 0.9 with T=0.5 -> 0.988, away from 0.5
    two = torch.sigmoid(torch.logit(conf[0:1].clamp(1e-6, 1 - 1e-6)) / 2.0)
    assert two[0, 0] < conf[0, 0]                  # 0.9 with T=2 -> 0.817, toward 0.5

    # 2. T = 1 is the identity, and the log keeps the raw stream
    vc._state["sts_t"] = torch.ones(GAMMA)
    assert torch.allclose(vc.calibrate(conf), torch.sigmoid(torch.logit(conf.clamp(1e-6, 1 - 1e-6))))
    vc.set_live_from_confidence(conf, 3)
    assert torch.equal(vc.live_buf()[:3], raw_live), "T=1 changed the rule"
    assert torch.equal(vc._state["last_conf"], conf), "the raw confidences were replaced"

    # 4. the loader validates
    import importlib

    def load_with(temperatures):
        os.environ["DSV41_VERIFY_CAP_STS"] = write_json(work / "sts.json", temperatures)
        snapshot = {k: v for k, v in vc._state.items() if k != "sts_t"}
        try:
            importlib.reload(vc)
        finally:
            vc._state.clear()
            vc._state.update(snapshot)
            os.environ.pop("DSV41_VERIFY_CAP_STS", None)

    for bad in ([-1.0] + [1.0] * (GAMMA - 1), [0.0] * GAMMA, [1.0] * (GAMMA + 1)):
        try:
            load_with(bad)
            raise SystemExit(f"the loader accepted {bad}")
        except AssertionError:
            pass

    # a single temperature broadcasts to every position
    load_with([1.5])
    fresh = sys.modules["verify_cap"]
    assert len(fresh._temps) == GAMMA and all(t == 1.5 for t in fresh._temps)

    print("test_verify_cap_sts: ok")


if __name__ == "__main__":
    main()