#!/usr/bin/env bash
# GPU-less sm_121a compile of the v41n native readers next to their stock twins, with ptxas -v and SASS checks.
#   bash tests/v41n_nvcc_check.sh OUT_DIR          (ds-b3-cutecheck:1: CUDA 13.0 nvcc; run with --network none)
# Writes OUT_DIR/ptxas-*.txt, OUT_DIR/sass-*.txt, OUT_DIR/summary.json. Fails on a compile error, any spill in a v41n
# kernel, a changed stock / io (EXTRA_FP4) decode kernel, or a generated prefill header that is not the generator's.
#
#   decode:  stock base sparse_mla_sm120_decode_dsv4.cu (25 + 5 kernels)  vs  v41c_decode_dsv4.cu (io, this tree and
#            942f0ec: SASS must be identical modulo symbol names)  vs  v41n_decode_dsv4.cu (native, converter warps:
#            25 kernels + the H16 / H64 K128 conversion-skipped timing diagnostics)
#   prefill: the production A twin (dual MG, BF16 QK, 16 heads, SWA top-k 128, 64-token extra pages, 1 head group)
#            vs  v41n_prefill.cu (the same kernel, FP4 pages of 128 / 256 records, 8 and 16 heads)
#   units:   one nope tile and one rope block of the converter alone (SASS instruction counts for the cost model)
#   binding: v41c_binding.cu syntax check against a stub of FlashInfer's tvm_ffi_utils.h (the image build compiles
#            it against the real one)
#   swan:    DSV41_SWA_NATIVE_DECODE. swan_decode_dsv4.cu (the stock kernel body of the vendored header, EXTRA_FP4 = false,
#            NATIVE_MODE = 0, at PAGE_BLOCK_SIZE 256) and its 64-token twin: the twin's SASS must equal the stock
#            base kernel's (modulo symbol names), and the 256-token kernels must have the twin's opcode sequence
#            (operands may differ: the page constant); the same opcode check for v41n's 256-token kernels
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; OUT="${1:?usage: v41n_nvcc_check.sh OUT_DIR}"; mkdir -p "$OUT"
F="$ROOT/runtime/flashinfer_v41c"
B="$ROOT/runtime/flashinfer_pce/base"
T=$(mktemp -d)
ARCH=(-gencode arch=compute_121a,code=sm_121a)
FLAGS=(-std=c++17 -O3 "${ARCH[@]}" --expt-relaxed-constexpr -Xptxas -v -c)
rc_all=0

# -- the 942f0ec io tree (base of this branch), for the identity check: V41N_BASE942 = an extracted
#    `git archive 942f0ec runtime/flashinfer_v41c` (containers without the repository), else git here
if [[ -n "${V41N_BASE942:-}" && -d "${V41N_BASE942}/runtime/flashinfer_v41c" ]]; then
  B942="${V41N_BASE942}/runtime/flashinfer_v41c"
elif git -C "$ROOT" cat-file -e 942f0ec 2>/dev/null; then
  mkdir -p "$T/base942"
  git -C "$ROOT" archive 942f0ec runtime/flashinfer_v41c | tar -x -C "$T/base942"
  B942="$T/base942/runtime/flashinfer_v41c"
else
  B942=""
fi

python3 "$F/gen_v41n_prefill.py" --check > "$OUT/gen-check.txt" 2>&1; gen=$?

nvcc "${FLAGS[@]}" -I "$B/include" "$B/csrc/sparse_mla_sm120_decode_dsv4.cu" -o "$T/stock_dec.o" > "$OUT/ptxas-stock-decode.txt" 2>&1; s_dec=$?
nvcc "${FLAGS[@]}" -I "$F/include" "$F/csrc/v41c_decode_dsv4.cu" -o "$T/io.o" > "$OUT/ptxas-io-decode.txt" 2>&1; s_io=$?
nvcc "${FLAGS[@]}" -I "$F/include" "$F/csrc/v41n_decode_dsv4.cu" -o "$T/nat_dec.o" > "$OUT/ptxas-v41n-decode.txt" 2>&1; s_nat=$?
s_io942=0
if [[ -n "$B942" ]]; then
  nvcc "${FLAGS[@]}" -I "$B942/include" "$B942/csrc/v41c_decode_dsv4.cu" -o "$T/io942.o" > "$OUT/ptxas-io-decode-942f0ec.txt" 2>&1; s_io942=$?
fi

# production A prefill twin (explicit instantiation of the stock kernel only)
cat > "$T/stock_prefill_twin.cu" <<'CU'
#include <flashinfer/attention/sparse_mla_sm120/prefill_kernel.cuh>
// prefill_kernel.cuh declares its kernels at global scope (the stock .cu includes it before its namespace)
const void* v41n_stock_prefill_twin() {
  return (const void*)&sparse_mla_prefill_mg_dual_kernel<ModelType::DSV4, ComputeMode::BF16, 16, 128, 64, 64, 1>;
}
CU
nvcc "${FLAGS[@]}" -I "$B/include" "$T/stock_prefill_twin.cu" -o "$T/stock_pf.o" > "$OUT/ptxas-stock-prefill-twin.txt" 2>&1; s_pf=$?
nvcc "${FLAGS[@]}" -I "$F/include" "$F/csrc/v41n_prefill.cu" -o "$T/nat_pf.o" > "$OUT/ptxas-v41n-prefill.txt" 2>&1; s_npf=$?
nvcc "${FLAGS[@]}" -I "$F/include" "$F/csrc/v41n_diag.cu" -o "$T/diag.o" > "$OUT/ptxas-v41n-diag.txt" 2>&1; s_diag=$?

# swan: the module's 256-token stock-body kernels and their 64-token twins (the twins are not in the module)
nvcc "${FLAGS[@]}" -I "$F/include" "$F/csrc/swan_decode_dsv4.cu" -o "$T/swan.o" > "$OUT/ptxas-swan-decode.txt" 2>&1; s_swan=$?
cat > "$T/swan_twin.cu" <<'CU'
#include <flashinfer_v41c/sparse_mla_sm120/decode_dsv4_kernel.cuh>
using namespace flashinfer::sparse_mla_sm120;
const void* swan_twins[] = {
  (const void*)&sparse_mla_decode_dsv4_kernel<ModelType::DSV4, 16, 128, 64, false, 0>,
  (const void*)&sparse_mla_decode_dsv4_kernel<ModelType::DSV4, 16, 192, 64, false, 0>,
  (const void*)&sparse_mla_decode_dsv4_kernel<ModelType::DSV4, 64, 128, 64, false, 0>,
  (const void*)&sparse_mla_decode_dsv4_kernel<ModelType::DSV4, 64, 192, 64, false, 0>,
};
CU
nvcc "${FLAGS[@]}" -I "$F/include" "$T/swan_twin.cu" -o "$T/swan_twin.o" > "$OUT/ptxas-swan-twin.txt" 2>&1; s_twin=$?

# converter units alone (SASS counts)
cat > "$T/units.cu" <<'CU'
#include <flashinfer_v41c/sparse_mla_sm120/v41n_convert.cuh>
using namespace flashinfer::sparse_mla_sm120;
__global__ void v41n_unit_nope(const uint32_t* __restrict__ in, const uint8_t* __restrict__ etab, uint32_t* out) {
  uint32_t w[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) w[i] = in[threadIdx.x * 9 + i];
  uint32_t o[16];
  const uint32_t e = v41n_nope_tile(w, in[threadIdx.x * 9 + 8], etab, o);
#pragma unroll
  for (int i = 0; i < 16; ++i) out[threadIdx.x * 17 + i] = o[i];
  out[threadIdx.x * 17 + 16] = e;
}
__global__ void v41n_unit_rope(const uint32_t* __restrict__ in, uint32_t* out) {
  uint32_t o[8];
  v41n_rope_block(in[threadIdx.x * 3], in[threadIdx.x * 3 + 1], in[threadIdx.x * 3 + 2], o);
#pragma unroll
  for (int i = 0; i < 8; ++i) out[threadIdx.x * 8 + i] = o[i];
}
CU
nvcc "${FLAGS[@]}" -I "$F/include" "$T/units.cu" -o "$T/units.o" > "$OUT/ptxas-units.txt" 2>&1; s_units=$?

sass() { cuobjdump -sass "$1" 2>/dev/null; }
[[ $s_dec == 0 ]] && sass "$T/stock_dec.o" > "$OUT/sass-stock-decode.txt"
[[ $s_io == 0 ]] && sass "$T/io.o" > "$OUT/sass-io-decode.txt"
[[ -n "$B942" && $s_io942 == 0 ]] && sass "$T/io942.o" > "$OUT/sass-io-decode-942f0ec.txt"
[[ $s_nat == 0 ]] && sass "$T/nat_dec.o" > "$OUT/sass-v41n-decode.txt"
[[ $s_pf == 0 ]] && sass "$T/stock_pf.o" > "$OUT/sass-stock-prefill-twin.txt"
[[ $s_npf == 0 ]] && sass "$T/nat_pf.o" > "$OUT/sass-v41n-prefill.txt"
[[ $s_units == 0 ]] && sass "$T/units.o" > "$OUT/sass-units.txt"
[[ $s_diag == 0 ]] && sass "$T/diag.o" > "$OUT/sass-v41n-diag.txt"
[[ $s_swan == 0 ]] && sass "$T/swan.o" > "$OUT/sass-swan-decode.txt"
[[ $s_twin == 0 ]] && sass "$T/swan_twin.o" > "$OUT/sass-swan-twin.txt"

# binding syntax check (stub TVM-FFI surface)
mkdir -p "$T/stub"
cat > "$T/stub/tvm_ffi_utils.h" <<'H'
#pragma once
#include <cstdint>
#include <cstddef>
#include <sstream>
#include <cuda_runtime.h>
namespace tvm { namespace ffi {
struct DLDataType { int code, bits, lanes; };
struct DLDevice { int device_type, device_id; };
struct TensorView {
  int ndim() const; int64_t size(int) const; int64_t stride(int) const; void* data_ptr() const;
  DLDataType dtype() const; DLDevice device() const;
};
template <typename T> struct Optional { bool has_value() const; const T& value() const; };
}}
using tvm::ffi::TensorView;
inline cudaStream_t get_stream(tvm::ffi::DLDevice) { return nullptr; }
struct V41nStubCheck { template <typename X> V41nStubCheck& operator<<(const X&) { return *this; } };
#define TVM_FFI_ICHECK(c) if (!(c)) V41nStubCheck()
#define TVM_FFI_ICHECK_EQ(a, b) TVM_FFI_ICHECK((a) == (b))
#define TVM_FFI_ICHECK_GE(a, b) TVM_FFI_ICHECK((a) >= (b))
#define TVM_FFI_DLL_EXPORT_TYPED_FUNC(name, fn) static auto v41n_stub_##name = &fn;
H
nvcc -std=c++17 "${ARCH[@]}" -I "$T/stub" -I "$F/include" -c "$F/csrc/v41c_binding.cu" -o "$T/binding.o" > "$OUT/binding-syntax.txt" 2>&1; s_bind=$?

python3 - "$OUT" "$T" "$gen" "$s_dec" "$s_io" "$s_io942" "$s_nat" "$s_pf" "$s_npf" "$s_units" "$s_bind" "${B942:+1}" "$s_diag" "$s_swan" "$s_twin" <<'PY'
import json, re, sys
out, tmp = sys.argv[1], sys.argv[2]
gen, s_dec, s_io, s_io942, s_nat, s_pf, s_npf, s_units, s_bind = map(int, sys.argv[3:12])
have942 = len(sys.argv) > 12 and sys.argv[12] == "1"
s_diag = int(sys.argv[13]) if len(sys.argv) > 13 else 1
s_swan = int(sys.argv[14]) if len(sys.argv) > 14 else 1
s_twin = int(sys.argv[15]) if len(sys.argv) > 15 else 1

def ptxas(path):
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
        m = re.search(r"used (\d+) barriers", ln)
        if m: rows[cur]["barriers"] = int(m.group(1))
        m = re.search(r"(\d+) bytes smem", ln)
        if m: rows[cur]["smem_static"] = int(m.group(1))
    return rows

def sass_funcs(path):
    """{function: [normalized instructions]} (addresses, encodings and symbol names dropped)."""
    funcs, cur = {}, None
    try:
        lines = open(path, errors="replace").read().splitlines()
    except OSError:
        return {}
    for ln in lines:
        m = re.match(r"\s+Function : (\S+)", ln)
        if m:
            cur = m.group(1); funcs[cur] = []; continue
        m = re.match(r"\s+/\*[0-9a-f]{4,}\*/\s+(.*?)\s*;?\s*(/\*.*\*/)?\s*$", ln)
        if cur and m:
            ins = m.group(1).strip().rstrip(";").strip()
            ins = re.sub(r"`\(\S+\)", "`(SYM)", ins)
            if ins:
                funcs[cur].append(ins)
    return funcs

def body_count(ins):
    return sum(1 for i in ins if not i.startswith("NOP") and not i.startswith("BRA 0x"))

def key_io(name):
    # strip the template tail that differs between 942f0ec (EXTRA_FP4) and this tree (EXTRA_FP4, EXTRA_NATIVE=false)
    m = re.search(r"sparse_mla_decode_dsv4_kernelIL9ModelType1ELi(\d+)ELi(\d+)ELi64ELb1E", name)
    if m: return ("decode", int(m.group(1)), int(m.group(2)))
    m = re.search(r"merge_kernelILi(\d+)E", name)
    if m: return ("merge", int(m.group(1)))
    return name

res = {"rc": {"gen_check": gen, "stock_decode": s_dec, "io_decode": s_io, "io_decode_942f0ec": s_io942,
              "v41n_decode": s_nat, "stock_prefill_twin": s_pf, "v41n_prefill": s_npf, "units": s_units,
              "binding_syntax": s_bind, "v41n_diag": s_diag, "swan_decode": s_swan, "swan_twin": s_twin}}
P = {k: ptxas(f"{out}/ptxas-{k}.txt") for k in ("stock-decode", "io-decode", "v41n-decode", "stock-prefill-twin",
                                                "v41n-prefill", "units", "v41n-diag", "swan-decode", "swan-twin")}
def summ(rows, pat):
    sel = {k: r for k, r in rows.items() if re.search(pat, k)}
    return {"kernels": len(sel), "regs": sorted({r.get("regs") for r in sel.values()}),
            "spill_bytes": sum(r.get("spill", 0) for r in sel.values()),
            "barriers": sorted({r.get("barriers") for r in sel.values() if r.get("barriers") is not None}),
            "smem_static": sorted({r.get("smem_static") for r in sel.values() if r.get("smem_static") is not None})}
res["ptxas"] = {
    "stock_decode": summ(P["stock-decode"], r"decode_dsv4_kernel"),
    "io_decode": summ(P["io-decode"], r"decode_dsv4_kernel"),
    "v41n_decode": summ(P["v41n-decode"], r"decode_dsv4_kernel"),
    "v41n_merge": summ(P["v41n-decode"], r"merge_kernel"),
    "stock_prefill_twin": summ(P["stock-prefill-twin"], r"prefill_mg_dual_kernel"),
    "v41n_prefill": summ(P["v41n-prefill"], r"v41n_prefill_mg_dual_kernel"),
    "units": {k: P["units"][k] for k in P["units"]},
    "v41n_diag": summ(P["v41n-diag"], r"v41n_convert_dump_kernel"),
    "swan_decode": summ(P["swan-decode"], r"decode_dsv4_kernel"),
    "swan_twin": summ(P["swan-twin"], r"decode_dsv4_kernel"),
    "v41n_decode_swan": summ(P["v41n-decode"], r"decode_dsv4_kernelIL9ModelType1ELi\d+ELi\d+ELi256ELb0ELi1E"),
}
S = {k: sass_funcs(f"{out}/sass-{k}.txt") for k in ("stock-decode", "io-decode", "io-decode-942f0ec", "v41n-decode",
                                                    "stock-prefill-twin", "v41n-prefill", "units", "v41n-diag",
                                                    "swan-decode", "swan-twin")}
# io identity: this tree's io kernels == 942f0ec's io kernels (instruction streams, symbol names normalized)
if have942 and S["io-decode-942f0ec"]:
    a = {key_io(k): v for k, v in S["io-decode"].items()}
    b = {key_io(k): v for k, v in S["io-decode-942f0ec"].items()}
    diff = sorted(str(k) for k in set(a) | set(b) if a.get(k) != b.get(k))
    res["io_sass_identical_to_942f0ec"] = {"kernels": len(b), "differ": diff[:8], "ok": not diff and len(a) == len(b)}
else:
    res["io_sass_identical_to_942f0ec"] = {"ok": None, "note": "942f0ec not available"}
# swan (DSV41_SWA_NATIVE_DECODE): (1) the vendored stock-body twins at 64-token pages == the stock base kernels
# (instruction streams, symbol names normalized); (2) every 256-token kernel (swan stock body, v41n native) has its
# 64-token twin's opcode sequence; the operand lines that differ are written out for review (the page constant)
def dkey(name):
    m = re.search(r"sparse_mla_decode_dsv4_kernelIL9ModelType1ELi(\d+)ELi(\d+)ELi(\d+)E(?:Lb(\d)ELi(\d)E)?", name)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4) or 0), int(m.group(5) or 0)) if m else None
def by_key(funcs):
    return {dkey(k): v for k, v in funcs.items() if dkey(k)}
def opcodes(ins):
    return [(i.split()[1] if i.startswith("@") else i.split()[0]) for i in ins]
stock_k, twin_k, swan_k, nat_k = (by_key(S["stock-decode"]), by_key(S["swan-twin"]), by_key(S["swan-decode"]),
                                  by_key(S["v41n-decode"]))
SWAN_SHAPES = [(16, 128), (16, 192), (64, 128), (64, 192)]
sw = {"twin_vs_stock": {}, "swan256_vs_twin64": {}, "v41n256_vs_v41n64": {}}
diff_lines = []
for h, k in SWAN_SHAPES:
    st, tw = stock_k.get((h, k, 64, 0, 0)), twin_k.get((h, k, 64, 0, 0))
    sw["twin_vs_stock"][f"h{h}_k{k}"] = bool(st) and st == tw
    for label, a, b in (("swan256_vs_twin64", swan_k.get((h, k, 256, 0, 0)), tw),
                        ("v41n256_vs_v41n64", nat_k.get((h, k, 256, 0, 1)), nat_k.get((h, k, 64, 0, 1)))):
        ok = bool(a) and bool(b) and opcodes(a) == opcodes(b)
        nd = [(i, x, y) for i, (x, y) in enumerate(zip(a or [], b or [])) if x != y]
        sw[label][f"h{h}_k{k}"] = {"same_opcodes": ok, "n_instr": [len(a or []), len(b or [])], "operand_diffs": len(nd)}
        diff_lines += [f"{label} h{h} k{k} @{i}: 256: {x}  |  64: {y}" for i, x, y in nd]
open(f"{out}/swan-operand-diffs.txt", "w").write("\n".join(diff_lines) + "\n")
sw["ok"] = (all(sw["twin_vs_stock"].values()) and len(sw["twin_vs_stock"]) == 4
            and all(v["same_opcodes"] for d in ("swan256_vs_twin64", "v41n256_vs_v41n64") for v in sw[d].values()))
res["swan"] = sw

def sizes(funcs, pat):
    return {re.sub(r"^_ZN10flashinfer16sparse_mla_sm120", "", k)[:90]: body_count(v)
            for k, v in funcs.items() if re.search(pat, k)}
res["sass_instructions"] = {
    "stock_decode_h64_k128": sizes(S["stock-decode"], r"decode_dsv4_kernelIL9ModelType1ELi64ELi128ELi64EEE"),
    "v41n_decode_h64_k128": sizes(S["v41n-decode"], r"decode_dsv4_kernelIL9ModelType1ELi64ELi128ELi64ELb0ELi1EE"),
    "stock_decode_h16_k128": sizes(S["stock-decode"], r"decode_dsv4_kernelIL9ModelType1ELi16ELi128ELi64EEE"),
    "v41n_decode_h16_k128": sizes(S["v41n-decode"], r"decode_dsv4_kernelIL9ModelType1ELi16ELi128ELi64ELb0ELi1EE"),
    "stock_prefill_twin": sizes(S["stock-prefill-twin"], r"prefill_mg_dual_kernel"),
    "v41n_prefill": sizes(S["v41n-prefill"], r"v41n_prefill_mg_dual_kernel"),
    "unit_nope_tile": sizes(S["units"], r"v41n_unit_nope"),
    "unit_rope_block": sizes(S["units"], r"v41n_unit_rope"),
}
# proxy fences (FENCE.VIEW.ASYNC.S): the stock kernels have one at init; the native kernels need one more, after the
# conversion stores, inside the chunk / tile loop (P1-1). Positions are instruction indices in the function.
def fences(funcs, pat):
    out = {}
    for k, v in funcs.items():
        if re.search(pat, k):
            pos = [i for i, ins in enumerate(v) if ins.startswith("FENCE.VIEW.ASYNC") or " FENCE.VIEW.ASYNC" in ins]
            bars = [i for i, ins in enumerate(v) if "SYNCS.ARRIVE" in ins or ins.startswith("BAR.") or "WARPSYNC" in ins]
            out[re.sub(r"^_ZN10flashinfer16sparse_mla_sm120", "", k)[:80]] = {"fence_positions": pos, "n_instr": len(v)}
    return out
res["proxy_fences"] = {
    "stock_decode_h64_k128": fences(S["stock-decode"], r"decode_dsv4_kernelIL9ModelType1ELi64ELi128ELi64EEE"),
    "v41n_decode_h64_k128": fences(S["v41n-decode"], r"decode_dsv4_kernelIL9ModelType1ELi64ELi128ELi64ELb0ELi1EE"),
    "v41n_decode_noconv_diag": fences(S["v41n-decode"], r"decode_dsv4_kernelIL9ModelType1ELi64ELi128ELi64ELb0ELi2EE"),
    "v41n_decode_noconv_diag_h16": fences(S["v41n-decode"], r"decode_dsv4_kernelIL9ModelType1ELi16ELi128ELi64ELb0ELi2EE"),
    "v41n_decode_h16_k128": fences(S["v41n-decode"], r"decode_dsv4_kernelIL9ModelType1ELi16ELi128ELi64ELb0ELi1EE"),
    "stock_prefill_twin": fences(S["stock-prefill-twin"], r"prefill_mg_dual_kernel"),
    "v41n_prefill_h16_p256": fences(S["v41n-prefill"], r"v41n_prefill_mg_dual_kernelIL9ModelType1EL11ComputeMode1ELi16ELi128ELi64ELi256"),
    "v41n_diag": fences(S["v41n-diag"], r"v41n_convert_dump_kernel"),
}
# instruction mix of the units (what the cost model counts)
mix = {}
for k, v in S["units"].items():
    c = {}
    for i in v:
        op = i.split()[0] if not i.startswith("@") else i.split()[1]
        op = op.split(".")[0]
        c[op] = c.get(op, 0) + 1
    mix[k] = dict(sorted(c.items(), key=lambda x: -x[1]))
res["unit_mix"] = mix
json.dump(res, open(f"{out}/summary.json", "w"), indent=1)
print(json.dumps({"rc": res["rc"], "ptxas": {k: v for k, v in res["ptxas"].items() if k != "units"},
                  "io_identity": res["io_sass_identical_to_942f0ec"], "sass": res["sass_instructions"],
                  "proxy_fences": res["proxy_fences"], "swan": res["swan"]}, indent=1))
ok = (all(v == 0 for v in res["rc"].values())
      and res["ptxas"]["v41n_decode"]["spill_bytes"] == 0 and res["ptxas"]["v41n_prefill"]["spill_bytes"] == 0
      and res["ptxas"]["v41n_decode"]["kernels"] == 31 and res["ptxas"]["v41n_prefill"]["kernels"] == 4
      and res["ptxas"]["swan_decode"]["kernels"] == 4 and res["ptxas"]["swan_decode"]["spill_bytes"] == 0
      and res["swan"]["ok"]
      and res["io_sass_identical_to_942f0ec"]["ok"] in (True, None)
      and all(len(f["fence_positions"]) >= 2 for k in ("v41n_decode_h64_k128", "v41n_decode_h16_k128",
                                                        "v41n_prefill_h16_p256")
              for f in res["proxy_fences"][k].values()))
sys.exit(0 if ok else 1)
PY
rc=$?
rm -rf "$T"
exit $rc
