"""Every b12x_next kernel class that receives the W4A8 N64 scale arguments, constructed with coded scales on and off
(needs nvidia-cutlass-dsl + torch; no GPU: the colima check image, or any CUDA host).

Fleet 2026-10-26 (REJECTIONS.md): every codes-on run died in `_get_dynamic_kernel` because MoEDynamicKernelSilu
(silu.py) spells out its keyword arguments and did not forward `w4a8_coded_scales`. A static check of one class plus
compile jobs of the phase kernels did not cover that constructor. This test drives the real construction paths:

1. signatures: every class in the kernel modules that takes the N64 scale switches (`w4a8_n64_repacked` /
   `n64_repacked`) must also take the coded switch (or **kwargs); the generic micro backends must take neither.
2. dynamic (activation silu, the production W4A8-MX N64 path): `_impl._get_dynamic_kernel` itself, stopped right
   after construction, for grouped routing at tiles 16 / 64 / 128 and direct routing at tile 16 (b12x's own limit),
   top-6 and top-3, deterministic output,
   codes on and off. Codes on: the kernel and its materialized phase 1 / phase 2 kernels carry the switch, or the
   monolithic guard refuses with its own message; the production plans (grouped, tiles 16 and 64) must construct.
3. micro (compact W4A8, decode): `_DirectW4A8CompactLaunch` carries the switch into the projection and phase 2 kernels;
   `_compiled_direct_w4a8_compact` and `_preparation._compact_launches` forward it.
4. micro formats (A/B retain mode): the launch keys the format on the tensors' dtype, refuses w13 / w2 disagreement
   and a prepared kernel compiled for the other format.

usage: PYTHONPATH=<runtime/b12x_next> python3 tests/test_scale_codes_kernels.py
"""
import inspect
import os
import sys

FAILS = []


def check(c, m):
    if not c:
        FAILS.append(m)


def test_signatures():
    from b12x_next.moe._shared.kernels import dynamic, micro, relu2, silu, situ
    from b12x_next.moe._shared.kernels import w4a8_compact_micro, w4a8_compact_projection, w4a8_phase1, w4a8_phase2
    seen = 0
    for mod in (dynamic, micro, relu2, silu, situ, w4a8_phase1, w4a8_phase2, w4a8_compact_projection, w4a8_compact_micro):
        for name, cls in inspect.getmembers(mod, inspect.isclass):
            if cls.__module__ != mod.__name__:
                continue
            try:
                params = inspect.signature(cls.__init__).parameters
            except (TypeError, ValueError):
                continue
            var_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())
            if "w4a8_n64_repacked" in params:
                seen += 1
                check("w4a8_coded_scales" in params or var_kw, f"{mod.__name__}.{name} takes w4a8_n64_repacked but not "
                                                               "w4a8_coded_scales")
            if "n64_repacked" in params or name in ("W4A8CompactMicroProjectionKernel", "_DirectW4A8CompactLaunch"):
                seen += 1
                check("coded_scales" in params or var_kw, f"{mod.__name__}.{name} takes the N64 switch but not coded_scales")
            if issubclass(cls, getattr(micro, "MoEMicroKernelBackend", ())) if hasattr(micro, "MoEMicroKernelBackend") \
                    else False:
                check("w4a8_n64_repacked" not in params, f"{name}: a generic micro backend takes N64 W4A8 scales")
    check(seen >= 6, f"only {seen} scale-receiving classes found")
    return seen


class _Built(Exception):
    def __init__(self, kernel):
        self.kernel = kernel


def test_dynamic():
    import torch
    from b12x_next.moe.fused_moe import _impl
    real_launch = _impl._DynamicMoEW4A8Launch

    def capture(kernel, **kw):
        raise _Built(kernel)

    _impl._DynamicMoEW4A8Launch = capture
    built = refused = 0
    try:
        for coded in (False, True):
            for direct in (False, True):
                for tile in (16, 64, 128):
                    for topk in (6, 3):
                        if direct and tile != 16:
                            continue                   # b12x: direct routing is M16-only (codes off alike)
                        tag = f"coded={coded} {'direct' if direct else 'grouped'} tile={tile} top{topk}"
                        try:
                            _impl._get_dynamic_kernel(
                                384, tile, 5120, 576, topk, tile * topk, topk_ids_dtype=torch.int32, fast_math=False,
                                activation="silu", quant_mode="w4a8_mx", w4a8_repacked=True, w4a8_n64_repacked=True,
                                w4a8_coded_scales=coded, direct_routing=direct, deterministic_output=True,
                                share_input_across_experts=True, swiglu_limit=10.0, planned_tile_m=tile,
                                planned_num_tokens=tile, mac_override=48)
                            FAILS.append(f"{tag}: construction was not reached")
                        except _Built as b:
                            k = b.kernel
                            built += 1
                            check(type(k).__name__ == "MoEDynamicKernelSilu", f"{tag}: built {type(k).__name__}")
                            check(getattr(k, "w4a8_coded_scales", None) is coded, f"{tag}: kernel switch "
                                                                                  f"{getattr(k, 'w4a8_coded_scales', None)}")
                            for ph in ("materialized_phase1_kernel", "materialized_phase2_kernel"):
                                sub = getattr(k, ph, None)
                                check(sub is not None and getattr(sub, "coded_scales", None) is coded,
                                      f"{tag}: {ph} switch {getattr(sub, 'coded_scales', None)}")
                            if coded:
                                check(k.external_materialized_fc1 and k.external_materialized_fc2,
                                      f"{tag}: coded scales reached a monolithic kernel")
                        except ValueError as exc:
                            if coded and "monolithic kernel" in str(exc):
                                refused += 1
                                check(not (not direct and tile in (16, 64)),
                                      f"{tag}: a production plan was refused: {exc}")
                            else:
                                FAILS.append(f"{tag}: {exc!r}"[:300])
                        except TypeError as exc:             # the fleet failure class
                            FAILS.append(f"{tag}: {exc!r}"[:300])
    finally:
        _impl._DynamicMoEW4A8Launch = real_launch
    check(built == 16, f"{built} dynamic kernels built, 16 expected (2 modes x (3 grouped + 1 direct tiles) x 2 top-k)")
    return built, refused


def test_micro():
    from b12x_next.moe._shared.kernels import w4a8_compact_micro as cm
    from b12x_next.moe.fused_moe import _preparation
    for coded in (False, True):
        kw = dict(max_tokens=8, num_topk=6, k=5120, n=576, experts=384, input_scale_count=384, down_scale_count=384,
                  swiglu_limit=10.0, fast_math=False, coded_scales=coded)
        launch = cm._DirectW4A8CompactLaunch(**kw)
        check(launch.coded_scales is coded and launch.projection.coded_scales is coded
              and launch.phase2.coded_scales is coded, f"compact micro coded={coded}: switches not forwarded")
    check("coded_scales" in inspect.signature(cm._compiled_direct_w4a8_compact.__wrapped__
                                              if hasattr(cm._compiled_direct_w4a8_compact, "__wrapped__")
                                              else cm._compiled_direct_w4a8_compact).parameters,
          "_compiled_direct_w4a8_compact does not take coded_scales")
    check("scale_codes_enabled()" in inspect.getsource(_preparation._compact_launches),
          "_compact_launches does not forward the switch")


def test_micro_formats():
    """A/B retain mode (2026-11-10): one process holds native and coded experts, so the compact micro launch keys the
    format on its tensors' dtype instead of the process-wide B12X_NEXT_W4A8_SCALE_CODES. It still refuses w13 / w2
    disagreement, and a prepared kernel (compiled by _compact_launches for the process format) of the other format.
    Driven on the CPU: the device-only helpers (scratch bytes, quantize, pointers, stream) are stubbed, the prepared
    kernel is a recorder."""
    import torch
    from b12x_next.moe._shared.kernels import w4a8_compact_micro as cm
    from b12x_next.moe.fused_moe import _preparation
    check("note_prepared_format(kernels, scale_codes_enabled())" in inspect.getsource(_preparation._compact_launches),
          "_compact_launches does not record its prepared kernels' format")
    saved = {n: getattr(cm, n) for n in ("_as_bytes", "_ptr", "current_cuda_stream")}
    cm._as_bytes = lambda t: t.view(torch.uint8).reshape(-1)
    cm._ptr = lambda *a, **k: None
    cm.current_cuda_stream = lambda: 0
    env = os.environ.pop("B12X_NEXT_W4A8_SCALE_CODES", None)
    try:
        cap, topk, k, n, e = 1, 2, 256, 64, 2
        scratch = torch.zeros(cm._layout(cap, topk, k, n)["total"][1], dtype=torch.uint8)

        def call(s13, s2, prepared):
            ran = []
            kern = (lambda *a: ran.append(len(a))) if prepared is not None else None
            if kern is not None:
                cm._PREPARED_FORMATS.pop(id(kern), None)
                if prepared != "unrecorded":
                    cm.note_prepared_format({torch.int32: kern}, prepared)
            try:
                cm.launch_w4a8_compact_micro(
                    scratch=scratch, a=torch.zeros(1, k, dtype=torch.bfloat16),
                    topk_ids=torch.zeros(1, topk, dtype=torch.int32), topk_weights=torch.zeros(1, topk),
                    w13=torch.zeros(e * 2 * n * k // 8, dtype=torch.int32), w13_scales=s13,
                    w2=torch.zeros(e * k * n // 8, dtype=torch.int32), w2_scales=s2,
                    alpha1=torch.ones(e), alpha2=torch.ones(e), input_scale=torch.ones(1), down_scale=torch.ones(1),
                    max_tokens=cap, num_topk=topk, swiglu_limit=None, fast_math=False,
                    _prepared_kernel=kern, _prepared_quantize=lambda *a: None)
            except ValueError as exc:
                return "refused" if "refused" in str(exc) else f"other: {exc}"
            return "ran" if ran else "not run"

        nat, cod = torch.zeros(8, dtype=torch.int32), torch.zeros(8, dtype=torch.uint8)
        for s13, s2, prepared, want in ((nat, nat, False, "ran"), (cod, cod, True, "ran"),
                                        (cod, cod, False, "refused"), (nat, nat, True, "refused"),
                                        (cod, nat, True, "refused"), (nat, cod, False, "refused"),
                                        (nat, nat, "unrecorded", "ran"), (cod, cod, "unrecorded", "refused")):
            got = call(s13, s2, prepared)
            check(got == want, f"compact micro w13 {s13.dtype} w2 {s2.dtype} prepared {prepared}: {got}, want {want}")
        os.environ["B12X_NEXT_W4A8_SCALE_CODES"] = "1"           # release mode: unrecorded kernels follow the switch
        check(call(cod, cod, "unrecorded") == "ran" and call(nat, nat, "unrecorded") == "refused",
              "unrecorded prepared kernels do not follow B12X_NEXT_W4A8_SCALE_CODES")
    finally:
        for n_, v in saved.items():
            setattr(cm, n_, v)
        os.environ.pop("B12X_NEXT_W4A8_SCALE_CODES", None)
        if env is not None:
            os.environ["B12X_NEXT_W4A8_SCALE_CODES"] = env


if __name__ == "__main__":
    os.environ.setdefault("CUTE_DSL_ARCH", "sm_121a")
    n = test_signatures()
    print(f"test_signatures: {'ok' if not FAILS else 'FAIL'} ({n} scale-receiving constructors)", flush=True)
    k = len(FAILS)
    built, refused = test_dynamic()
    print(f"test_dynamic: {'ok' if len(FAILS) == k else 'FAIL'} ({built} built, {refused} monolithic refusals)", flush=True)
    k = len(FAILS)
    test_micro()
    print(f"test_micro: {'ok' if len(FAILS) == k else 'FAIL'}", flush=True)
    k = len(FAILS)
    test_micro_formats()
    print(f"test_micro_formats: {'ok' if len(FAILS) == k else 'FAIL'}", flush=True)
    for f in FAILS:
        print("  " + f)
    print("test_scale_codes_kernels: " + ("ok" if not FAILS else f"{len(FAILS)} failure(s)"))
    sys.exit(1 if FAILS else 0)
