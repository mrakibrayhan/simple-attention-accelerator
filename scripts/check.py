"""
check.py - verifies one simulated test case.

Usage (from the repo root, after run_sim.py):   python scripts/check.py hand2

Checks
  1. BIT-EXACT: every hardware buffer (Q, K, V, S, weights) and the output O
     against the bit-exact Python model  -> must be 0 mismatches
  2. ACCURACY:  hardware O against the floating-point reference -> max / mean error
  3. SANITY:    softmax rows sum to ~1;  latency equals the analytic formula
Returns exit code 0 on pass, 1 on fail.
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model"))
from fixed_model import read_hex  # noqa: E402


def expected_cycles(N, D, DK):
    return 3 * N * DK * (D + 1) + N * (N * (DK + 1) + 4 * N + 35 + DK * (N + 1)) + 1


def check(name, verbose=True):
    d = os.path.join(ROOT, "tests", name)
    N, D, DK = map(int, open(os.path.join(d, "params.txt")).read().split())
    ok = True
    log = []

    # 1. bit-exact comparison of every stage
    stages = [("Q", "hw_q.hex", True), ("K", "hw_k.hex", True), ("V", "hw_v.hex", True),
              ("S", "hw_s.hex", True), ("A", "hw_a.hex", False), ("O", "output.hex", True)]
    for st, hw_file, signed in stages:
        hw = read_hex(os.path.join(d, hw_file), signed=signed)
        ex = read_hex(os.path.join(d, f"expected_{st.lower()}.hex"), signed=signed)
        bad = [n for n in range(max(len(hw), len(ex)))
               if n >= len(hw) or n >= len(ex) or hw[n] != ex[n]]
        if bad:
            ok = False
            n = bad[0]
            log.append(f"  {st}: FAIL {len(bad)} mismatches, first at index {n}: "
                       f"hw={hw[n] if n < len(hw) else None} expected={ex[n] if n < len(ex) else None}")
        else:
            log.append(f"  {st}: bit-exact ({len(hw)} values)")

    # 2. accuracy vs float
    o_hw = np.array(read_hex(os.path.join(d, "output.hex"))).reshape(N, DK) / 256.0
    o_fl = np.loadtxt(os.path.join(d, "expected_float.txt")).reshape(N, DK)
    err = np.abs(o_hw - o_fl)
    rel = err.max() / max(np.abs(o_fl).max(), 1e-9)

    # 3. sanity
    a = np.array(read_hex(os.path.join(d, "hw_a.hex"), signed=False)).reshape(N, N) / 65536.0
    rowsum_err = np.abs(a.sum(axis=1) - 1).max()
    cyc = int(open(os.path.join(d, "cycles.txt")).read())
    cyc_ok = cyc == expected_cycles(N, D, DK)
    ok &= cyc_ok

    res = dict(name=name, N=N, D=D, DK=DK, ok=ok, max_err=err.max(), mean_err=err.mean(),
               max_lsb=err.max() * 256, rel=rel, rowsum_err=rowsum_err, cycles=cyc)
    if verbose:
        print(f"[{name}]  N={N} D={D} DK={DK}")
        print("\n".join(log))
        print(f"  accuracy vs float : max |err| = {err.max():.5f} ({err.max()*256:.2f} LSB), "
              f"mean = {err.mean():.5f}, max/|O|max = {rel*100:.2f} %")
        print(f"  softmax row sums  : max |sum-1| = {rowsum_err:.2e}")
        print(f"  latency           : {cyc} cycles (formula {expected_cycles(N, D, DK)}) "
              f"{'OK' if cyc_ok else 'MISMATCH'}")
        print(f"  RESULT            : {'PASS' if ok else 'FAIL'}")
    return res


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(0 if check(sys.argv[1])["ok"] else 1)
