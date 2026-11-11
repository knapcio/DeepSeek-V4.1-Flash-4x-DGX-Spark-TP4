#!/usr/bin/env python3
"""Static SASS gate for the evict-first copies: value flow of every global-memory descriptor (stdlib only).

On sm_12x every global access names a 64-bit memory descriptor ``desc[URd]`` (a uniform register pair URd:URd+1). A
stock access reads the default descriptor the kernel loads from the constant bank (``LDCU.64 URd, c[0x0][0x358]``);
an ``L2::cache_hint`` copy reads the ``createpolicy`` value instead (evict_first, fraction 1.0 =
0x12f0000000000000). ptxas folded a uniform shared offset into the hinted ``LDGSTS`` and built it in the low half of
the descriptor pair (or in a pair it never wrote), which faulted on GB10 with an illegal instruction.

Per function this check builds the control-flow graph of the listing (``BRA`` targets, conditional branches,
``EXIT``), runs a forward constant propagation over the uniform registers (UMOV / USHF / ULOP3 / UIADD3 / constant-bank
loads exact, every other write = defined but unknown unless a source register is undefined on some path: then the
result is undefined too (taint), a predicated write = the meet of both, kernel entry = every register undefined) and then classifies each access that names ``desc[URd]``:

  default   URd:URd+1 holds c[0x0][0x358]:c[0x0][0x35c] on every path
  policy    URd:URd+1 holds 0x00000000:0x12f00000 on every path (only an LDGSTS may)
  bad       anything else: an odd pair, a half that is undefined on some path (use before write), overwritten by
            another value on some path, or unknown; or a uniform register in the address that is undefined on
            some path

  python3 tests/sass_desc_check.py SASS.txt [--expect phase1=8,phase2=8,frontend=0]   exit 0 clean, 1 bad / count
  python3 tests/sass_desc_check.py --selftest
``--expect``: the required number of policy LDGSTS per function class (phase1 / phase2 / frontend / other); a class
not named must have none. Used by tests/cutecheck_moe_evict_first.py on every job.
"""
import re
import sys

POLICY = (("k", 0x0), ("k", 0x12F00000))
DEFAULT = (("c", 0, 0x358), ("c", 0, 0x35C))
UNDEF, TOP = ("U",), ("T",)
NREG = 64
INS = re.compile(r"^\s*/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;\s*(?:/\*.*)?$")
PRED = re.compile(r"^(@!?U?P(?:T|\d+))\s+(.*)$")
UR = re.compile(r"^UR(\d+)$")
UPRED = re.compile(r"^!?UP(?:T|\d+)$")
DESC = re.compile(r"desc\[UR(\d+)\]")
CBANK = re.compile(r"^c\[(0x[0-9a-f]+)\]\[(0x[0-9a-f]+)\]$")
MASK = 0xFFFFFFFF


def fclass(name):
    return ("phase1" if "Phase1Kernel" in name else "phase2" if "Phase2Kernel" in name
            else "frontend" if "MoEDynamicKernel" in name else "other")


def split_ops(s):
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def parse(section):
    """[(addr, pred, opcode, [operands], text)] in listing order."""
    out = []
    for ln in section.splitlines():
        m = INS.match(ln)
        if not m:
            continue
        body = m.group(2).strip()
        pred = None
        mp = PRED.match(body)
        if mp:
            pred, body = mp.group(1), mp.group(2)
            if pred in ("@PT", "@UPT"):
                pred = None
        op, _, rest = body.partition(" ")
        out.append((int(m.group(1), 16), pred, op, split_ops(rest), body))
    return out


def meet(a, b):
    if a is None:
        return b
    if b is None or a == b:
        return a
    if a == UNDEF or b == UNDEF:
        return UNDEF
    return TOP


def ival(tok, st):
    """Value of a source operand: ("k", int) / ("c", ...) / UNDEF / TOP."""
    if tok == "URZ" or tok == "RZ":
        return ("k", 0)
    m = UR.match(tok)
    if m:
        return st[int(m.group(1))]
    if re.match(r"^-?0x[0-9a-f]+$", tok):
        return ("k", int(tok, 16) & MASK)
    if re.match(r"^-?\d+$", tok):
        return ("k", int(tok) & MASK)
    return TOP


def lop3(a, b, c, lut):
    r = 0
    for i in range(8):
        if lut >> i & 1:
            r |= (a if i & 4 else ~a) & (b if i & 2 else ~b) & (c if i & 1 else ~c)
    return r & MASK


def evaluate(op, ops, st):
    """(dest_reg, [values]) for a uniform-register write, else None."""
    if not ops:
        return None
    m = UR.match(ops[0])
    if not m:
        return None                                   # URZ, a predicate, a memory operand, a vector register
    d = int(m.group(1))
    base, mods = op.split(".")[0], op.split(".")[1:]
    width = 4 if "128" in mods else 2 if ("64" in mods or "WIDE" in mods) else 1
    src = [o for o in ops[1:] if not UPRED.match(o)]
    # uninitialized-source taint (astra batch 4): a value computed from a register that is undefined on some path is
    # itself undefined on that path (never "defined, unknown"), so an address or descriptor built from it fails
    regs = [int(x) for o in src for x in re.findall(r"\bUR(\d+)\b", o)]
    if width != 1 and "WIDE" not in mods:
        regs += [r + 1 for r in regs]
    if any(r < NREG and st[r] in (UNDEF, None) for r in regs):
        return d, [UNDEF] * width
    v = TOP
    try:
        if base in ("LDCU", "ULDC") and len(src) == 1 and CBANK.match(src[0]):
            bank, off = (int(x, 16) for x in CBANK.match(src[0]).groups())
            return d, [("c", bank, off + 4 * i) for i in range(width)]
        if width != 1:
            return d, [TOP] * width
        vals = [ival(s, st) for s in src]
        if base == "UMOV" and len(vals) == 1:
            v = vals[0]
        elif base == "USHF" and len(vals) == 3 and all(x[0] == "k" for x in vals) and vals[2] == ("k", 0) \
                and "U32" in mods and ("L" in mods or "R" in mods):
            a, s = vals[0][1], vals[1][1] & 31
            v = ("k", (a << s) & MASK if "L" in mods else a >> s)
        elif base == "ULOP3" and "LUT" in mods and len(vals) == 4 and all(x[0] == "k" for x in vals):
            v = ("k", lop3(vals[0][1], vals[1][1], vals[2][1], vals[3][1]))
        elif base == "UIADD3" and len(vals) == 3 and all(x[0] == "k" for x in vals):
            v = ("k", sum(x[1] for x in vals) & MASK)
    except (IndexError, ValueError):
        v = TOP
    return d, [v]


def blocks(ins):
    """Basic blocks over the listing: (start index, end index exclusive, successor start addresses)."""
    addr_idx = {a: i for i, (a, *_rest) in enumerate(ins)}
    leaders = {0}
    for i, (a, pred, op, ops, _) in enumerate(ins):
        base = op.split(".")[0]
        if base in ("BRA", "EXIT", "RET", "BRX", "JMP", "JMX", "CALL"):
            if i + 1 < len(ins):
                leaders.add(i + 1)
            if base == "BRA":
                t = int(ops[-1], 16) if ops and re.match(r"^0x[0-9a-f]+$", ops[-1]) else None
                if t is None or t not in addr_idx:
                    raise ValueError(f"branch at {a:#x} to an unknown target: {ops}")
                leaders.add(addr_idx[t])
            elif base in ("BRX", "JMP", "JMX", "CALL", "RET"):
                raise ValueError(f"unsupported control flow at {a:#x}: {op}")
    starts = sorted(leaders)
    out = {}
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else len(ins)
        a, pred, op, ops, _ = ins[e - 1]
        base = op.split(".")[0]
        succ = []
        cond = pred is not None or (base == "BRA" and len(ops) > 1)
        if base == "BRA":
            succ.append(addr_idx[int(ops[-1], 16)])
            if cond and e < len(ins):
                succ.append(e)
        elif base == "EXIT":
            if cond and e < len(ins):
                succ.append(e)
        elif e < len(ins):
            succ.append(e)
        out[s] = (e, succ)
    return out


def transfer(ins, s, e, st, visit=None):
    st = list(st)
    for i in range(s, e):
        a, pred, op, ops, text = ins[i]
        if visit:
            visit(i, st)
        w = evaluate(op, ops, st)
        if w is None:
            continue
        d, vals = w
        for j, v in enumerate(vals):
            if d + j >= NREG:
                continue
            st[d + j] = meet(st[d + j], v) if pred else v
    return st


def check_function(section):
    """{'policy': n, 'default': n, 'bad': [..], 'ldgsts': n} for one function's listing."""
    ins = parse(section)
    res = {"policy": 0, "default": 0, "bad": [], "ldgsts": 0}
    if not ins:
        return res
    try:
        bb = blocks(ins)
    except ValueError as exc:
        res["bad"].append(f"cfg: {exc}")
        return res
    IN = {s: None for s in bb}
    IN[0] = [UNDEF] * NREG
    work = [0]
    while work:
        s = work.pop()
        e, succ = bb[s]
        out = transfer(ins, s, e, IN[s])
        for t in succ:
            new = out if IN[t] is None else [meet(x, y) for x, y in zip(IN[t], out)]
            if new != IN[t]:
                IN[t] = new
                work.append(t)

    def visit(i, st):
        a, pred, op, ops, text = ins[i]
        dm = DESC.search(text)
        if not dm:
            return
        is_ldgsts = op.startswith("LDGSTS")
        res["ldgsts"] += is_ldgsts
        d = int(dm.group(1))
        mem = next((o for o in ops if o.startswith("[")), "")
        offs = [int(x) for x in re.findall(r"UR(\d+)", mem)]
        why = []
        if d % 2:
            why.append(f"odd descriptor pair UR{d}")
        pair = (st[d], st[d + 1]) if d + 1 < NREG else (UNDEF, UNDEF)
        if any(st[o] in (UNDEF, None) for o in offs):
            why.append(f"address register {['UR%d' % o for o in offs if st[o] in (UNDEF, None)]} undefined on a path")
        kind = "policy" if pair == POLICY else "default" if pair == DEFAULT else None
        if kind is None:
            why.append(f"descriptor UR{d}:UR{d + 1} = {pair}")
        elif kind == "policy" and not is_ldgsts:
            why.append(f"policy descriptor on a non-LDGSTS {op}")
        if why:
            res["bad"].append(f"{a:#06x} {text[:90]} :: {'; '.join(why)}")
        elif is_ldgsts:
            res[kind] += 1

    for s, (e, _succ) in bb.items():
        if IN[s] is not None:
            transfer(ins, s, e, IN[s], visit)
    return res


def check_text(sass):
    """[(function name, class, result)] for a cuobjdump -sass listing."""
    out = []
    for sec in re.split(r"\n\s*Function : ", "\n" + sass)[1:]:
        name = sec.split("\n", 1)[0].strip()
        out.append((name, fclass(name), check_function(sec)))
    return out


def gate(sass, expect):
    """(ok, lines): no bad access anywhere and the policy LDGSTS count per class == expect (absent class = 0)."""
    rows = check_text(sass)
    lines, ok = [], bool(rows)
    got = {}
    for name, cls, r in rows:
        got[cls] = got.get(cls, 0) + r["policy"]
        lines.append(f"{cls:8s} ldgsts {r['ldgsts']:3d} policy {r['policy']:3d} default {r['default']:3d} "
                     f"bad {len(r['bad'])} {name[-48:]}")
        lines += [f"    BAD {b}" for b in r["bad"][:6]]
        ok &= not r["bad"]
    for cls in set(got) | set(expect):
        if got.get(cls, 0) != expect.get(cls, 0):
            ok = False
            lines.append(f"    COUNT {cls}: {got.get(cls, 0)} policy LDGSTS, required {expect.get(cls, 0)}")
    if not rows:
        lines.append("    no function in the listing")
    return ok, lines


# ------------------------------------------------------------------------------------------------------ selftest
def _fn(name, body):
    lines = [f"\t\tFunction : {name}"]
    for i, t in enumerate(body):
        lines.append(f"        /*{i * 16:04x}*/                   {t} ;      /* 0x0000000000000000 */")
    return "\n".join(lines) + "\n"


def selftest():
    pol = ["UMOV UR24, 0xf0", "USHF.R.U32 UR24, UR24, 0x4, URZ", "UMOV UR25, 0x120",
           "ULOP3.LUT UR25, UR25, 0xf, UR24, 0xf8, !UPT", "USHF.L.U32 UR25, UR25, 0x14, URZ", "UMOV UR24, URZ"]
    dflt = ["LDCU.64 UR4, c[0x0][0x358]"]
    hint = "LDGSTS.E.BYPASS.LTC128B.128 [R77+0x840], desc[UR24][R8.64]"
    stock = "LDGSTS.E.BYPASS.128 [R72], desc[UR4][R22.64]"
    P1 = "kernel_W4A8MaterializedPhase1Kernel_object_at_x"
    cases = [
        ("clean: policy + default", dflt + pol + [hint, stock, "EXIT"], {"phase1": 1}, True),
        ("count: one hinted copy short", dflt + pol + [hint, stock, "EXIT"], {"phase1": 2}, False),
        ("count: hinted copy in an off kernel", dflt + pol + [hint, "EXIT"], {}, False),
        # the 4.8.0 miscompile: the policy pair's low word overwritten with the shared-stage offset
        ("overwritten low word", dflt + pol + ["S2UR UR10, SR_CgaCtaId", "UMOV UR18, UR24", "UMOV UR19, UR25",
                                                "UMOV UR18, UR10", "LDGSTS.E.BYPASS.128 [R14+UR18+0x1400], desc[UR19][R2.64]",
                                                "EXIT"], {"phase1": 1}, False),
        # the 4.6.3 miscompile: a pair and an offset never written in the function
        ("never written", dflt + ["LDGSTS.E.BYPASS.128 [R77+UR0+0x840], desc[UR2][R8.64]", "EXIT"], {"phase1": 1}, False),
        # astra's synthetic: the descriptor is written, but after its use on the loop's first trip (use before init)
        ("use before init in a loop", dflt + ["UMOV UR30, 0x0", hint.replace("UR24", "UR26")] + pol
         + ["UMOV UR26, UR24", "UMOV UR27, UR25", "@P0 BRA 0x10", "EXIT"], {"phase1": 1}, False),
        ("defined on one branch only", dflt + ["@P1 BRA 0x80"] + pol + [hint, "EXIT"], {"phase1": 1}, False),
        ("predicated overwrite", dflt + pol + ["@P2 UMOV UR24, 0x400", hint, "EXIT"], {"phase1": 1}, False),
        ("written on both branches", dflt + ["@P1 BRA 0x90"] + pol + ["BRA 0xf0"] + pol + [hint, "EXIT"],
         {"phase1": 1}, True),
        ("wrong policy value (evict_last)", dflt + [p.replace("0x120", "0x100") for p in pol] + [hint, "EXIT"],
         {"phase1": 1}, False),
        ("odd pair", dflt + pol + ["UMOV UR25, UR24", "UMOV UR26, 0x12f00000",
                                   "LDGSTS.E.BYPASS.128 [R1], desc[UR25][R2.64]", "EXIT"], {"phase1": 1}, False),
        ("default pair clobbered", dflt + ["UMOV UR4, 0x0", stock, "EXIT"], {}, False),
        ("policy on a plain load", dflt + pol + ["LDG.E.128 R4, desc[UR24][R2.64]", "EXIT"], {}, False),
        # astra batch 4: an uninitialized UR20 laundered through UIADD3 into the address register of 8 hinted copies
        ("astra4: UR20 -> UIADD3 UR18 -> 8 hinted LDGSTS", dflt + pol + ["UIADD3 UR18, UR20, 0x400, URZ"]
         + [f"LDGSTS.E.BYPASS.128 [R77+UR18+{0x840 + 0x800 * i:#x}], desc[UR24][R8.64]" for i in range(8)] + ["EXIT"],
         {"phase1": 8}, False),
        ("taint through a chain (ULEA, USEL)", dflt + pol + ["ULEA UR6, UR21, UR4, 0x4", "USEL UR7, UR6, URZ, UP0",
                                                              "LDGSTS.E.BYPASS.128 [R1+UR7], desc[UR24][R8.64]", "EXIT"],
         {"phase1": 1}, False),
        ("taint into the descriptor's high word", dflt + pol + ["UIADD3 UR25, UR25, UR22, URZ", hint, "EXIT"],
         {"phase1": 1}, False),
        ("a defined offset from defined inputs is fine", dflt + pol + ["S2UR UR20, SR_CgaCtaId", "UIADD3 UR18, UR20, 0x400, URZ",
                                                                       "LDGSTS.E.BYPASS.128 [R77+UR18+0x840], desc[UR24][R8.64]",
                                                                       "EXIT"], {"phase1": 1}, True),
        ("64-bit op reading an undefined partner", dflt + pol + ["S2UR UR20, SR_CgaCtaId", "UIADD3.64 UR18, UR20, URZ, URZ",
                                                                 "LDGSTS.E.BYPASS.128 [R1+UR18], desc[UR24][R8.64]", "EXIT"],
         {"phase1": 1}, False),
    ]
    bad = []
    for name, body, expect, want in cases:
        ok, lines = gate(_fn(P1, body), expect)
        if ok != want:
            bad.append(f"{name}: {ok}, want {want} :: {lines}")
    for b in bad:
        print("FAIL", b)
    print("sass_desc_check selftest:", "ALL PASS" if not bad else f"{len(bad)} FAIL")
    return 1 if bad else 0


def main(argv):
    if argv[:1] == ["--selftest"]:
        return selftest()
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    expect = {}
    if "--expect" in argv:
        i = argv.index("--expect")
        expect = {k: int(v) for k, v in (x.split("=") for x in argv[i + 1].split(",") if x)}
        argv = argv[:i] + argv[i + 2:]
    rc = 0
    for p in argv:
        ok, lines = gate(open(p, errors="replace").read(), expect)
        print(f"== {p}: {'OK' if ok else 'BAD'}")
        print("\n".join(lines))
        rc |= not ok
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
