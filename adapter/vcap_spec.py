"""DSV41_VERIFY_CAP conf: specs, parsed without torch (adapter/verify_cap.py).

  "conf:T"              one threshold for every batch size
  "conf:T/4:T4/8:T8"    a threshold per batch size: each "/<bs>:<thr>" entry applies from that batch size up
                        (ascending, 2..64), the leading value below the first entry; an entry that repeats the
                        leading value closes a band (conf:0.1/4:0.25/8:0.1 = 0.25 at bs 4-7 only)
Every threshold may also be the per-position form "t2,t3,t4,t5" ([0.0, t2, t3, t4, t5]: draft 1 is always
verified). Without a "/" a spec parses exactly as the single-threshold parser always did.
"""
MAX_BS = 64


def one_threshold(text):
    vals = [float(v) for v in text.split(",")]
    return vals[0] if len(vals) == 1 else [0.0] + vals


def parse_conf_spec(spec):
    """'conf:<thr>[/<bs>:<thr>...]' -> (default threshold, ((min_bs, threshold), ...) ascending)."""
    spec = (spec or "").strip()
    if not spec.startswith("conf:"):
        raise ValueError(f"DSV41_VERIFY_CAP={spec!r} is not a conf: policy")
    parts = spec.split(":", 1)[1].split("/")
    default = one_threshold(parts[0])
    table = []
    for item in parts[1:]:
        bs_text, sep, thr_text = item.partition(":")
        if not sep or not thr_text:
            raise ValueError(f"DSV41_VERIFY_CAP: '{item}' is not <bs>:<threshold>")
        min_bs = int(bs_text)
        if not 2 <= min_bs <= MAX_BS or (table and min_bs <= table[-1][0]):
            raise ValueError(f"DSV41_VERIFY_CAP: batch sizes must ascend within 2..{MAX_BS} ({spec!r})")
        table.append((min_bs, one_threshold(thr_text)))
    return default, tuple(table)


def threshold_for(parsed, bs):
    """The threshold a parsed spec gives a step of this batch size."""
    default, table = parsed
    for min_bs, thr in reversed(table):
        if bs >= min_bs:
            return thr
    return default


def per_bs(spec):
    """The threshold of every batch size 1..MAX_BS (what a spec does, independent of how it is spelled)."""
    parsed = parse_conf_spec(spec)
    return [threshold_for(parsed, bs) for bs in range(1, MAX_BS + 1)]


def bands(parsed):
    """Human-readable bands 'bs 1-3 0.1 | bs 4-7 0.25 | bs 8-64 0.1' (the boot receipt)."""
    out, start, cur = [], 1, threshold_for(parsed, 1)
    for bs in range(2, MAX_BS + 2):
        t = threshold_for(parsed, bs) if bs <= MAX_BS else None
        if bs > MAX_BS or t != cur:
            out.append(f"bs {start}-{bs - 1} {cur}")
            start, cur = bs, t
    return " | ".join(out)
