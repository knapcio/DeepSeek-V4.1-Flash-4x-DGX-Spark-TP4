#!/usr/bin/env bash
# GPU-less sm_121a compile of the DSV41_PREFILL_NATIVE_EXTRA module, generated from the vendored FlashInfer base, next
# to the stock prefill, with ptxas -v.
#   bash tests/pce_nvcc_check.sh OUT_DIR        (ds-b3-cutecheck:1: CUDA 13.0 nvcc, --network none)
# Fails on a generator anchor error, a compile error, a missing PBSX 128/256 dual kernel, or a pce kernel whose
# registers / shared memory / spill bytes differ from the stock PBSX 64 kernel of the same shape (the template argument
# changes address arithmetic only; two stock dual kernels already spill 8 B). Writes OUT_DIR/ptxas-{stock,pce}.txt and OUT_DIR/summary.json.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; OUT="${1:?usage: pce_nvcc_check.sh OUT_DIR}"; mkdir -p "$OUT"
F="$ROOT/runtime/flashinfer_pce"
T=$(mktemp -d); mkdir -p "$T/gen"
python3 "$F/gen_pce_sources.py" "$F/base/csrc" "$F/base/include" "$T/gen" > "$OUT/gen.txt" 2>&1 || { cat "$OUT/gen.txt"; rm -rf "$T"; exit 1; }
cp "$T/gen/manifest.json" "$OUT/manifest.json"
FLAGS=(-std=c++17 -O3 -DNDEBUG -gencode arch=compute_121a,code=sm_121a --expt-relaxed-constexpr -Xptxas -v -c)
FLAGS_C=(-std=c++17 -O3 -DNDEBUG -gencode arch=compute_121a,code=sm_121a --expt-relaxed-constexpr -c)
INC=(-I "$F/base/include" -I "$ROOT/tests/pce_stub")
nvcc "${FLAGS[@]}" "${INC[@]}" "$F/base/csrc/sparse_mla_sm120_prefill.cu" -o "$T/stock.o" > "$OUT/ptxas-stock.txt" 2>&1; s=$?
nvcc "${FLAGS[@]}" "${INC[@]}" "$T/gen/pce_prefill.cu" -o "$T/pce.o" > "$OUT/ptxas-pce.txt" 2>&1; p=$?
nvcc "${FLAGS_C[@]}" "${INC[@]}" "$T/gen/pce_paged_attention.cu" -o "$T/orc.o" > "$OUT/orchestrator.txt" 2>&1; o=$?
nvcc "${FLAGS_C[@]}" "${INC[@]}" "$T/gen/pce_binding.cu" -o "$T/bind.o" > "$OUT/binding.txt" 2>&1; b=$?
python3 - "$OUT" "$s" "$p" "$o" "$b" <<'PY'
import json, re, subprocess, sys
out, s, p, o, b = sys.argv[1], *map(int, sys.argv[2:6])
def parse(path):
    rows, cur = {}, None
    for ln in open(path, errors="replace"):
        m = re.search(r"Compiling entry function '(\S+)'", ln)
        if m:
            cur = m.group(1); rows[cur] = {}
        if cur is None:
            continue
        m = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", ln)
        if m: rows[cur]["spill"] = int(m.group(1)) + int(m.group(2))
        m = re.search(r"Used (\d+) registers", ln)
        if m: rows[cur]["regs"] = int(m.group(1))
        m = re.search(r"(\d+) bytes smem", ln)
        if m: rows[cur]["smem"] = int(m.group(1))
    return rows
st, pc = parse(f"{out}/ptxas-stock.txt"), parse(f"{out}/ptxas-pce.txt")
# dual kernels mangle as ...ELi<PBS=64>ELi<PBSX>ELi<NHG>EE...; the PBSX 64 twin is the same name with PBSX -> 64
PAT = re.compile(r"ELi64ELi(\d+)ELi(\d)EE")
pce_dual = {n: r for n, r in pc.items() if "dual" in n and PAT.search(n)}
pbsx = sorted({PAT.search(n).group(1) for n in pce_dual})
mism = []
for n, r in pce_dual.items():
    twin = PAT.sub(lambda m: f"ELi64ELi64ELi{m.group(2)}EE", n, count=1)
    sr = st.get(twin)
    if sr is None:
        mism.append({"kernel": n, "why": "no stock PBSX 64 twin"}); continue
    if (r.get("regs"), r.get("smem"), r.get("spill", 0)) != (sr.get("regs"), sr.get("smem"), sr.get("spill", 0)):
        mism.append({"kernel": n, "pce": r, "stock64": sr})
summ = {"stock_rc": s, "pce_rc": p, "orchestrator_rc": o, "binding_rc": b,
        "stock_kernels": len(st), "pce_kernels": len(pc), "pce_dual_kernels": len(pce_dual),
        "pce_non_dual_kernels": len(pc) - len(pce_dual), "pce_pbsx": pbsx,
        "pce_spills": sum(r.get("spill", 0) for r in pc.values()),
        "stock64_twin_spills": sum(st[PAT.sub(lambda m: f"ELi64ELi64ELi{m.group(2)}EE", n, count=1)].get("spill", 0)
                                   for n in pce_dual if PAT.sub(lambda m: f"ELi64ELi64ELi{m.group(2)}EE", n, count=1) in st),
        "pce_max_regs": max([r.get("regs", 0) for r in pc.values()] or [0]),
        "regs_smem_vs_stock64_mismatch": mism}
json.dump(summ, open(f"{out}/summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in summ.items() if k != "regs_smem_vs_stock64_mismatch"}),
      "mismatches", len(mism))
ok = (s == p == o == b == 0 and summ["pce_dual_kernels"] == 20 and summ["pce_non_dual_kernels"] == 0
      and pbsx == ["128", "256"] and summ["pce_spills"] == summ["stock64_twin_spills"] and not mism)
sys.exit(0 if ok else 1)
PY
rc=$?
rm -rf "$T"
exit $rc
