"""Capture the target hidden states the DSpark draft is trained against. Default OFF.

WHY THIS HOOK AND NOT THE OBVIOUS ONE
-------------------------------------
The brief proposes editing `DSparkWorkerV2._forward_decode` to insert ~15 lines directly before
`dspark_worker_v2.py:883` (`logits_output.hidden_states = None`). That is the right *place*, but it
needs a source edit inside a 250-line method, which the adapter mechanism cannot express and which
would have to be re-checked against every engine bump.

`TargetVerifyExecutor.commit_hidden` (`dspark_verify.py:316`) is called from exactly that point --
the `if not folded_commit:` block that ends one line before the null -- and it receives everything
the tap needs as named arguments. Wrapping it is a plain function wrapper with no source edit.

WHAT IT MISSES, STATED UP FRONT
-------------------------------
`commit_hidden` is skipped when `folded_commit` is True, which requires `fold_eligible`, which
requires `sampling_info.is_all_greedy` (`dspark_worker_v2.py:774-784`). So the tap is BLIND to
all-greedy batches. Production runs temperature=1.0/top_p=0.95 (`ds_agent_proxy.py:110-118`), so the
folded path is disabled for essentially all live traffic and the tap sees it all. But a corpus
generated deliberately at T=0 on an otherwise idle server would be captured NOT AT ALL. Generate any
self-made corpus at T=1.0 -- which is also the distribution the draft is actually serving under, so
it is the right training data regardless.

THE TENSOR IS NOT SAFE TO COPY ASYNCHRONOUSLY
---------------------------------------------
`logits_output.hidden_states` is a slice of the target-verify CUDA graph's persistent output buffer
(`full_cuda_graph_backend.py:143-161` returns the same object on every replay; the decode runner
merely slices it). Being a slice of a static buffer is precisely why it does NOT survive the next
replay: the next step's replay writes those same addresses on the compute stream, with no ordering
against a side stream. So we clone on the CURRENT stream first and only then hand the clone to a
non-blocking device-to-host copy on a dedicated stream, releasing the clone when the event fires.
This is the same pattern the in-tree `kv_canary/runner/future_tensor.py:52-56` uses, with the same
comment ("Must happen in current stream, not d2h stream").

WHAT THE TOP-K IS, AND WHAT IT IS NOT
-------------------------------------
`logits_output.next_token_logits` has already been mutated in place by the verify logits adjustments
before this point (`dspark_verify.py:281-287`), and by the grammar mask when a grammar is live
(`dspark_worker_v2.py:816-825`). So the stored top-k is a POST-adjustment distribution, not the raw
target logits, and a distillation loss built on it is fitting that adjusted distribution. Verify what
`apply_dflash_verify_logits_adjustments` actually does under temperature=1.0/top_p=0.95 before
relying on the values. The stored `argmax` is safe either way: temperature scaling and top-p masking
are both monotone on the surviving support, so they cannot move the arg of the max, and the argmax is
exactly the label the greedy acceptance rule compares against.

NO DEVICE SYNC, EVER
--------------------
Nothing here calls `.item()`, `.cpu()` (blocking), or `torch.cuda.synchronize()` on the hot path. A
single sync would add a full device round trip to a ~55 ms step. Row filtering by `commit_lens`
needs that tensor on the host, so we do NOT filter on device: all `verify_num_draft_tokens` rows are
shipped along with `commit_lens`, and the rejected rows are dropped by the offline reader.

ENV
---
  DSV41_DRAFT_CAPTURE=1              arm the hook at all (default off; absent => zero overhead)
  DSV41_DRAFT_CAPTURE_TRIGGER        file whose presence enables capture, default /state/CAPTURE_DRAFT
                                     (create it to start, delete it to stop -- no restart either way)
  DSV41_DRAFT_CAPTURE_OUT            output directory, default /state/draft-capture
  DSV41_DRAFT_CAPTURE_EVERY          capture 1 step in N, default 1
  DSV41_DRAFT_CAPTURE_TOPK           target top-k logits to keep per row, default 64, 0 disables
  DSV41_DRAFT_CAPTURE_FP8            store hidden as float8_e4m3fn, default 1 (halves the bytes)
  DSV41_DRAFT_CAPTURE_MAX_GIB        stop capturing after this many GiB, default 200
  DSV41_DRAFT_CAPTURE_BENCH=1        log per-step hook cost instead of writing anything
"""
import os
import queue
import threading
import time

ENABLED = os.environ.get("DSV41_DRAFT_CAPTURE", "0").strip() not in ("0", "", "off", "false")
TRIGGER = os.environ.get("DSV41_DRAFT_CAPTURE_TRIGGER", "/state/CAPTURE_DRAFT")
OUT_DIR = os.environ.get("DSV41_DRAFT_CAPTURE_OUT", "/state/draft-capture")
EVERY = int(os.environ.get("DSV41_DRAFT_CAPTURE_EVERY", "1"))
# Override site: DSV41_DRAFT_CAPTURE_EVERY below 1 used to be clamped to 1 silently. A user-set
# value below 1 now raises when the capture is enabled.
if EVERY < 1:
    if ENABLED:
        raise RuntimeError(f"DSV41_DRAFT_CAPTURE_EVERY={EVERY} must be >= 1 (DSV41_DRAFT_CAPTURE is set)")
    EVERY = 1
TOPK = int(os.environ.get("DSV41_DRAFT_CAPTURE_TOPK", "64"))
USE_FP8 = os.environ.get("DSV41_DRAFT_CAPTURE_FP8", "1").strip() not in ("0", "", "off", "false")
MAX_BYTES = int(float(os.environ.get("DSV41_DRAFT_CAPTURE_MAX_GIB", "200")) * (1 << 30))
BENCH = os.environ.get("DSV41_DRAFT_CAPTURE_BENCH", "0").strip() not in ("0", "", "off", "false")
DEBUG = os.environ.get("DSV41_DRAFT_CAPTURE_DEBUG", "0").strip() not in ("0", "", "off", "false")

_S = {
    "calls": 0, "captured": 0, "bytes": 0, "armed": False, "stopped": False,
    "skip_armed": 0, "skip_rank": 0, "skip_graph": 0, "skip_kwargs": 0,
    "skip_compact": 0, "skip_hidden": 0, "skip_stopped": 0, "queued": 0,
    "stream": None, "pending": [], "shard": 0, "rows": 0, "last_check": 0.0,
    "hook_ns": 0, "hook_n": 0, "rank": None,
}


def _is_rank0():
    """Capture on ONE rank only.

    enable_dp_attention=False, dp_size=1, enable_layernorm_sp=False, so the hidden states are
    REPLICATED across tensor-parallel ranks, not sharded by sequence. Every rank holds a
    bit-identical copy. Without this gate the tap would write four identical corpora and four
    times the disk traffic. The in-tree block-accept recorder gates the same way
    (`dspark_block_accept_estimator.py:822`, `if tp_rank != 0: return None`).

    Resolved lazily: the distributed group does not exist yet when the adapter installs.
    """
    if _S["rank"] is None:
        r = -1
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                r = dist.get_rank()
        except Exception:
            pass
        if r < 0:
            for k in ("RANK", "LOCAL_RANK", "SGLANG_TP_RANK"):
                v = os.environ.get(k)
                if v is not None and v.strip().lstrip("-").isdigit():
                    r = int(v)
                    break
        _S["rank"] = r if r >= 0 else 0
        if _S["rank"] != 0:
            print(f"[draft_capture] rank {_S['rank']} is not 0, staying idle", flush=True)
    return _S["rank"] == 0


def _armed():
    """Re-stat the trigger at most once a second so arming costs nothing on the hot path."""
    now = time.monotonic()
    if now - _S["last_check"] > 1.0:
        _S["last_check"] = now
        _S["armed"] = os.path.exists(TRIGGER)
    return _S["armed"]


_Q = queue.Queue(maxsize=64)


def _writer_loop(torch):
    """torch.save is synchronous disk I/O; it must not run on the scheduler thread."""
    while True:
        rec = _Q.get()
        if rec is None:
            return
        try:
            path = os.path.join(_S.get("dir") or OUT_DIR, f"shard-{_S['shard']:06d}.pt")
            tmp = path + ".tmp"
            torch.save(rec, tmp)
            os.replace(tmp, path)
            _S["shard"] += 1
            _S["rows"] += int(rec["hidden"].shape[0]) * int(rec["stride"])
            _S["bytes"] += int(rec["hidden"].numel() * rec["hidden"].element_size())
            if _S["bytes"] >= MAX_BYTES and not _S["stopped"]:
                _S["stopped"] = True
                print(f"[draft_capture] reached {MAX_BYTES/(1<<30):.1f} GiB, stopping", flush=True)
        except Exception as exc:
            # Override site: a writer-thread error used to stop the capture for the rest of the
            # run with one print. The error is now parked in _S["writer_error"] and the next
            # commit_hidden call raises it, because DSV41_DRAFT_CAPTURE no longer takes effect.
            print(f"[draft_capture] writer error, stopping: {exc!r}", flush=True)
            _S["writer_error"] = repr(exc)
            _S["stopped"] = True


def _drain(torch, force=False):
    """Hand finished device->host copies to the writer thread. One step late, by construction."""
    keep = []
    for item in _S["pending"]:
        if not force and not item["event"].query():
            keep.append(item)
            continue
        item["event"].synchronize()
        # _hold pins the DEVICE clones until the copy lands; it must never reach torch.save.
        rec = {k: v for k, v in item.items() if k not in ("event", "_hold")}
        try:
            _Q.put_nowait(rec)
            _S["queued"] += 1
        except queue.Full:
            # Disk cannot keep up. Dropping a shard is strictly better than stalling decode.
            # Override site: a full writer queue drops the shard. The fall-through is kept and
            # logged as WARNING on the 1st, 101st, 201st ... drop.
            _S["dropped"] = _S.get("dropped", 0) + 1
            if _S["dropped"] % 100 == 1:
                print(f"WARNING [draft_capture] writer queue full, dropped {_S['dropped']} shards", flush=True)
    _S["pending"] = keep


def _capture(torch, batch, logits_output, commit_lens, bs, verify_num_draft_tokens, out_tokens=None,
             verify_ids=None, draft_logits=None, draft_temps=None):
    t0 = time.perf_counter_ns() if BENCH else 0
    hidden = logits_output.hidden_states
    # Override site: a verify step whose logits_output carries no hidden states is not captured.
    # The fall-through is kept and logged once.
    if hidden is None:
        _S["skip_hidden"] += 1
        if _S["skip_hidden"] == 1:
            print("WARNING [draft_capture] logits_output.hidden_states is None; verify step not captured", flush=True)
        return
    if _S["stream"] is None:
        _S["stream"] = torch.cuda.Stream()

    h = hidden.view(bs, verify_num_draft_tokens, -1)
    # 1. Clone on the CURRENT stream. The source is a CUDA-graph static buffer and the next
    #    replay overwrites it; a copy issued on another stream would race that replay.
    src = h.to(torch.float8_e4m3fn) if USE_FP8 else h
    dev_hidden = src.clone()
    dev_commit = commit_lens.clone()
    # Committed tokens, [bs, stride] int64 from BuildOutTokens: columns 0..correct_len-1 are the
    # accepted draft tokens, column correct_len is the bonus, so [:, :commit_len] is exactly the
    # sequence that got committed. This is the teacher-forcing input the Markov head needs
    # (prev_token for capture row j>=1 is out_tokens[j-1]); `argmax` is only the target's mode and
    # differs from the sampled token on ~23% of prose rows (measured 2026-09-22, mean p_top1 0.77).
    # In the folded-accept path this is a view of a graph buffer, hence the clone on the current stream.
    dev_out = None if out_tokens is None else out_tokens[:bs].to(torch.int32).clone()
    dev_vid = None if verify_ids is None else verify_ids[:bs].to(torch.int32).clone()
    dev_tmp = None if draft_temps is None else draft_temps[:bs].float().clone()
    dev_dv = dev_di = dev_dt = None
    if draft_logits is not None and TOPK > 0:
        dl = draft_logits[:bs].float()                       # [bs, 5, V] markov-corrected draft logits
        dv, di = torch.topk(dl, TOPK, dim=-1)
        # the draft's normaliser over the FULL vocab, so q can be rebuilt exactly for the top-64
        lse = torch.logsumexp(dl / (draft_temps[:bs].float().view(-1, 1, 1) if draft_temps is not None else 1.0), dim=-1)
        dev_dv, dev_di, dev_dt = dv.to(torch.float16).clone(), di.to(torch.int32).clone(), lse.clone()
    logits = logits_output.next_token_logits
    dev_topv = dev_topi = dev_argmax = None
    if logits is not None:
        lg = logits.view(bs, verify_num_draft_tokens, -1)
        dev_argmax = lg.argmax(dim=-1).clone()      # the ground truth the draft must match
        if TOPK > 0:
            v, i = torch.topk(lg, TOPK, dim=-1)     # distillation target; done on device, never shipped whole
            dev_topv, dev_topi = v.to(torch.float16).clone(), i.to(torch.int32).clone()

    # 2. Non-blocking D2H on a side stream, ordered after the clone.
    ev_ready = torch.cuda.Event()
    ev_ready.record()
    done = torch.cuda.Event()
    with torch.cuda.stream(_S["stream"]):
        _S["stream"].wait_event(ev_ready)
        rec = {
            "hidden": dev_hidden.to("cpu", non_blocking=True),
            "commit_lens": dev_commit.to("cpu", non_blocking=True),
            "argmax": None if dev_argmax is None else dev_argmax.to("cpu", non_blocking=True),
            "topk_val": None if dev_topv is None else dev_topv.to("cpu", non_blocking=True),
            "topk_idx": None if dev_topi is None else dev_topi.to("cpu", non_blocking=True),
            "out_tokens": None if dev_out is None else dev_out.to("cpu", non_blocking=True),
            "verify_ids": None if dev_vid is None else dev_vid.to("cpu", non_blocking=True),
            "draft_topk_val": None if dev_dv is None else dev_dv.to("cpu", non_blocking=True),
            "draft_topk_idx": None if dev_di is None else dev_di.to("cpu", non_blocking=True),
            "draft_lse_t": None if dev_dt is None else dev_dt.to("cpu", non_blocking=True),
            "draft_temps": None if dev_tmp is None else dev_tmp.to("cpu", non_blocking=True),
        }
        done.record(_S["stream"])
    # 3. Hold references to the device clones until the copy completes, or they are freed early.
    rec["_hold"] = (dev_hidden, dev_commit, dev_argmax, dev_topv, dev_topi, dev_out, dev_vid, dev_dv, dev_di, dev_dt, dev_tmp)
    rec["event"] = done
    rec["forward_ct"] = int(getattr(batch, "forward_iter", -1))
    rec["rids"] = [getattr(r, "rid", None) for r in getattr(batch, "reqs", [])[:bs]]
    rec["bs"] = int(bs)
    rec["stride"] = int(verify_num_draft_tokens)
    rec["fp8"] = bool(USE_FP8)
    _S["pending"].append(rec)
    _S["captured"] += 1
    if BENCH:
        _S["hook_ns"] += time.perf_counter_ns() - t0
        _S["hook_n"] += 1
        if _S["hook_n"] % 200 == 0:
            print(f"[draft_capture] hook mean {_S['hook_ns']/_S['hook_n']/1e6:.3f} ms "
                  f"over {_S['hook_n']} steps", flush=True)


def _run_dir(torch):
    """One directory per engine run. The shard counter restarts at 0 on every boot, so a shared
    directory silently overwrote the previous session's data (found 2026-09-20: a fresh boot
    clobbered shard-000000..000322 of a 4732-shard capture)."""
    import time as _t
    d = os.path.join(OUT_DIR, _t.strftime("run-%Y%m%d-%H%M%S") + "-%d" % os.getpid())
    os.makedirs(d, exist_ok=True)
    meta = {
        "started": _t.strftime("%Y-%m-%dT%H:%M:%S%z"), "pid": os.getpid(), "topk": TOPK,
        "fp8": USE_FP8, "every": EVERY, "max_gib": MAX_BYTES / (1 << 30),
        "folded_sampling": os.environ.get("SGLANG_DSPARK_FOLDED_SAMPLING", ""),
        "block_size": os.environ.get("DSPARK_BLOCK_SIZE", ""),
        "out_tokens": True,   # schema v3: shards carry the committed tokens, not only argmax
        "schema": 4,          # v4: + verify_ids, draft top-64 logits, draft logsumexp at the draft temperature
    }
    try:
        import json as _j
        with open(os.path.join(d, "meta.json"), "w") as f:
            _j.dump(meta, f, indent=1)
    except Exception:
        pass
    return d


def install(module):
    """module = sglang.srt.speculative.dspark_components.dspark_verify"""
    if not ENABLED:
        return
    import torch
    cls = getattr(module, "TargetVerifyExecutor", None)
    # Override site: a missing TargetVerifyExecutor used to skip the install silently. The gate
    # DSV41_DRAFT_CAPTURE is set when this runs, so the gate that does not take effect now raises.
    if cls is None:
        raise RuntimeError(
            f"DSV41_DRAFT_CAPTURE={os.environ.get('DSV41_DRAFT_CAPTURE', '')!r} is set but "
            f"{getattr(module, '__name__', module)} has no TargetVerifyExecutor")
    if getattr(cls, "_dsv41_draft_capture", False):
        return
    cls._dsv41_draft_capture = True
    os.makedirs(OUT_DIR, exist_ok=True)
    _S["dir"] = _run_dir(torch)
    threading.Thread(target=_writer_loop, args=(torch,), daemon=True,
                     name="dsv41-draft-capture-writer").start()
    orig = cls.commit_hidden
    orig_accept = getattr(cls, "accept_and_finalize", None)

    if orig_accept is not None:
        # The worker calls accept_and_finalize() and then commit_hidden() on the same executor in the
        # same step (dspark_worker_v2.py:837-887); commit_hidden never receives out_tokens, so park the
        # accept result on the instance for the capture below. Attribute only, no device work here.
        def wrapped_accept(self, *a, **kw):
            outs = orig_accept(self, *a, **kw)
            try:
                self._dsv41_out_tokens = getattr(outs, "out_tokens", None)
                # v4: the drafted tokens (verify rows 1..5 inputs) and the draft's own distribution,
                # so block-level verification rules can be evaluated offline on engine-exact p and q
                self._dsv41_verify_ids = kw.get("verify_ids_2d")
                db = kw.get("draft_block")
                self._dsv41_draft_logits = getattr(db, "corrected_logits", None)
                self._dsv41_draft_temps = getattr(db, "temperatures", None)
            except Exception:
                pass
            return outs
        cls.accept_and_finalize = wrapped_accept

    def wrapped(self, *a, **kw):
        _S["calls"] += 1
        if _S.get("writer_error"):
            raise RuntimeError(
                f"DSV41_DRAFT_CAPTURE={os.environ.get('DSV41_DRAFT_CAPTURE', '')!r} is set but the capture "
                f"writer thread failed ({_S['writer_error']})")
        try:
            if _S["stopped"]:
                _S["skip_stopped"] += 1
            elif not _is_rank0():
                _S["skip_rank"] += 1
            elif not _armed():
                _S["skip_armed"] += 1
            elif torch.cuda.is_current_stream_capturing():
                _S["skip_graph"] += 1
            elif _S["calls"] % EVERY != 0:
                pass
            else:
                bs = kw.get("bs")
                lo = kw.get("logits_output")
                cl = kw.get("commit_lens")
                # Override sites: a commit_hidden call without bs / logits_output / commit_lens, or
                # a compact verify run, is not captured. Each fall-through is kept and logged once.
                if bs is None or lo is None or cl is None:
                    _S["skip_kwargs"] += 1
                    if _S["skip_kwargs"] == 1:
                        print("WARNING [draft_capture] commit_hidden call without bs/logits_output/commit_lens "
                              "is not captured", flush=True)
                elif kw.get("run_compact"):
                    _S["skip_compact"] += 1
                    if _S["skip_compact"] == 1:
                        print("WARNING [draft_capture] compact verify run (run_compact) is not captured", flush=True)
                else:
                    _capture(torch, kw.get("batch"), lo, cl, int(bs),
                             int(self.verify_num_draft_tokens),
                             out_tokens=getattr(self, "_dsv41_out_tokens", None),
                             verify_ids=getattr(self, "_dsv41_verify_ids", None),
                             draft_logits=getattr(self, "_dsv41_draft_logits", None),
                             draft_temps=getattr(self, "_dsv41_draft_temps", None))
            if DEBUG and _S["calls"] % 100 == 0:
                print("[draft_capture] calls=%d captured=%d queued=%d shards=%d pending=%d | skip: armed=%d rank=%d graph=%d kwargs=%d compact=%d hidden=%d stopped=%d" % (
                    _S["calls"], _S["captured"], _S["queued"], _S["shard"], len(_S["pending"]),
                    _S["skip_armed"], _S["skip_rank"], _S["skip_graph"], _S["skip_kwargs"],
                    _S["skip_compact"], _S["skip_hidden"], _S["skip_stopped"]), flush=True)
            if _S["pending"]:
                _drain(torch)
        except Exception as exc:
            # Override site: an error in the capture path used to stop the capture for the rest of
            # the run with one print. The gate DSV41_DRAFT_CAPTURE is set when this runs, so the
            # gate that does not take effect now raises.
            _S["stopped"] = True
            raise RuntimeError(
                f"DSV41_DRAFT_CAPTURE={os.environ.get('DSV41_DRAFT_CAPTURE', '')!r} is set but the capture "
                f"failed ({exc!r})") from exc
        return orig(self, *a, **kw)

    cls.commit_hidden = wrapped
    print(f"[draft_capture] armed (v4, out_tokens + draft q) on TargetVerifyExecutor.commit_hidden; trigger={TRIGGER} "
          f"out={_S['dir']} every={EVERY} topk={TOPK} fp8={USE_FP8}", flush=True)
