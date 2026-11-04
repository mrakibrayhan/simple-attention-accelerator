"""
run_sim.py - compile and simulate ONE test case with Icarus Verilog.

Usage (from the repo root):
    python scripts/run_sim.py hand2
    python scripts/run_sim.py rand_n4_d8_dk4
    python scripts/run_sim.py rand_n16_d64_dk64 --novcd   (skip the ~110 MB VCD)

It reads N D DK from tests/<name>/params.txt, compiles with
    iverilog -g2012 -P tb_attention.N=.. -P tb_attention.D=.. -P tb_attention.DK=..
then runs vvp inside tests/<name>/ so the testbench finds its .hex files and
writes output.hex / output.txt / attention.vcd there.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RTL = ["attention_top.v", "mac.v", "round_sat.v", "exp2_unit.v", "exp2_lut.v", "recip_div.v"]


def run_sim(name, quiet=False, vcd=True):
    tdir = os.path.join(ROOT, "tests", name)
    with open(os.path.join(tdir, "params.txt")) as f:
        N, D, DK = f.read().split()
    os.makedirs(os.path.join(ROOT, "build"), exist_ok=True)
    vvp = os.path.join(ROOT, "build", f"{name}.vvp")
    cmd = ["iverilog", "-g2012", "-o", vvp,
           "-P", f"tb_attention.N={N}", "-P", f"tb_attention.D={D}", "-P", f"tb_attention.DK={DK}",
           os.path.join(ROOT, "tb", "tb_attention.v")] + [os.path.join(ROOT, "rtl", r) for r in RTL]
    if not quiet:
        print(" ".join(os.path.relpath(c, ROOT) if os.path.isabs(c) else c for c in cmd))
    subprocess.run(cmd, check=True)
    out = subprocess.run(["vvp", "-n", vvp] + ([] if vcd else ["+novcd"]), cwd=tdir, check=True, capture_output=True, text=True).stdout
    if not quiet:
        print(out.strip())
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    run_sim(sys.argv[1], vcd="--novcd" not in sys.argv)
