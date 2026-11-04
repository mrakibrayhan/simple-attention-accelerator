"""
run_all.py - full regression in one command (from the repo root):

    python scripts/run_all.py

  1. regenerates the exp LUT (rtl/exp2_lut.v) and all test vectors
  2. runs the unit tests  (tb_mac, tb_exp_div)
  3. simulates every test in tests/ and checks it (bit-exact + accuracy + latency)
  4. writes results/summary.md
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from run_sim import run_sim  # noqa: E402
from check import check      # noqa: E402

BIG = 50_000   # tests with more MAC operations than this run without a VCD


def sh(cmd, cwd=ROOT):
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout


def main():
    py = sys.executable
    print("== 1. generate LUT + test vectors"); print(sh([py, "model/gen_lut.py"]).strip())
    print(sh([py, "model/gen_tests.py"]).strip())

    print("\n== 2. unit tests")
    os.makedirs(os.path.join(ROOT, "build"), exist_ok=True)
    unit_ok = True
    for tb, srcs in [("tb_mac", ["rtl/mac.v", "rtl/round_sat.v"]),
                     ("tb_exp_div", ["rtl/exp2_unit.v", "rtl/exp2_lut.v", "rtl/recip_div.v"])]:
        sh(["iverilog", "-g2012", "-o", f"build/{tb}.vvp", f"tb/{tb}.v"] + srcs)
        out = sh(["vvp", "-n", f"{tb}.vvp"], cwd=os.path.join(ROOT, "build"))
        line = [l for l in out.splitlines() if "PASS" in l or "FAIL" in l][-1]
        unit_ok &= "PASS" in line
        print("  " + line)

    print("\n== 3. system tests")
    names = sorted(n for n in os.listdir(os.path.join(ROOT, "tests"))
                   if os.path.isfile(os.path.join(ROOT, "tests", n, "params.txt")))
    size = lambda n: (lambda N, D, DK: 3 * N * D * DK + 2 * N * N * DK)(
        *map(int, open(os.path.join(ROOT, "tests", n, "params.txt")).read().split()))
    names.sort(key=size)                     # smallest first
    results = []
    for n in names:
        N, D, DK = map(int, open(os.path.join(ROOT, "tests", n, "params.txt")).read().split())
        run_sim(n, quiet=True, vcd=3 * N * D * DK + 2 * N * N * DK < BIG)
        results.append(check(n))
        print()

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "summary.md"), "w") as f:
        f.write("| Test | N | D | d_k | Bit-exact vs model | Max error vs float | Mean error | Cycles | @100 MHz |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['name']} | {r['N']} | {r['D']} | {r['DK']} | {'PASS' if r['ok'] else 'FAIL'} "
                    f"| {r['max_err']:.4f} ({r['max_lsb']:.2f} LSB) | {r['mean_err']:.5f} "
                    f"| {r['cycles']:,} | {r['cycles']/100:.2f} µs |\n")
    all_ok = unit_ok and all(r["ok"] for r in results)
    print("== summary written to results/summary.md")
    print("== OVERALL:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
