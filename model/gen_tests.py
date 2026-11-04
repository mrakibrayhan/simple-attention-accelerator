"""
gen_tests.py - builds every test case under tests/<name>/

For each test it:
  1. writes (or reuses) human-readable DECIMAL inputs  inputs/x.txt, wq.txt, wk.txt, wv.txt
  2. quantizes them to Q8.8 and writes the $readmemh files  x.hex wq.hex wk.hex wv.hex
  3. writes params.txt  ("N D DK")
  4. runs the bit-exact model and writes expected_*.hex (q,k,v,s,a,o)
  5. runs the float model and writes expected_float.txt

Run from the repo root:  python model/gen_tests.py
"""
import os
import numpy as np
from fixed_model import (attention_fixed, attention_float, to_fixed, write_hex,
                         read_txt_matrix)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(ROOT, "tests")

# Step-1 hand example (the one you solved on paper)
HAND = dict(
    x=[[1, 0, 1, 0], [0, 2, 0, 2]],
    wq=[[.5, 0, .5, 0], [0, .5, 0, .5], [.5, 0, 0, .5], [0, .5, .5, 0]],
    wk=[[0, .5, 0, .5], [.5, 0, .5, 0], [0, .5, .5, 0], [.5, 0, 0, .5]],
    wv=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
)

# (name, N tokens, D model dim, DK head dim, seed, input range)
RANDOM_TESTS = [
    ("rand_n4_d8_dk4",    4,  8,  4, 1, 1.0),
    ("rand_n8_d16_dk16",  8, 16, 16, 2, 1.0),
    ("rand_n16_d64_dk64", 16, 64, 64, 3, 1.0),   # realistic head size (d_k = 64, as in BERT/GPT-2)
    ("sharp_n8_d16_dk4",  8, 16,  4, 4, 3.0),   # large scores -> near one-hot softmax, exp flush-to-zero
]


def write_txt(path, M):
    with open(path, "w") as f:
        for r in M:
            f.write(" ".join(f"{v:.6g}" for v in r) + "\n")


def random_inputs(N, D, DK, seed, rng_x=1.0):
    rng = np.random.default_rng(seed)
    q = lambda a: np.round(a * 256) / 256          # snap to the Q8.8 grid
    x = q(rng.uniform(-rng_x, rng_x, (N, D)))
    w = lambda: q(rng.normal(0, 1.5 / np.sqrt(D), (D, DK)))
    return dict(x=x.tolist(), wq=w().tolist(), wk=w().tolist(), wv=w().tolist())


def build(name, mats=None):
    d = os.path.join(TESTS, name)
    inp = os.path.join(d, "inputs")
    os.makedirs(inp, exist_ok=True)
    if mats is not None:                            # write decimal inputs
        for k, M in mats.items():
            write_txt(os.path.join(inp, f"{k}.txt"), M)
    M = {k: read_txt_matrix(os.path.join(inp, f"{k}.txt")) for k in ("x", "wq", "wk", "wv")}
    fx_in = {k: [[to_fixed(v) for v in r] for r in m] for k, m in M.items()}
    for k, m in fx_in.items():
        write_hex(os.path.join(d, f"{k}.hex"), m)
    N, D, DK = len(M["x"]), len(M["x"][0]), len(M["wq"][0])
    with open(os.path.join(d, "params.txt"), "w") as f:
        f.write(f"{N} {D} {DK}\n")

    fx = attention_fixed(fx_in["x"], fx_in["wq"], fx_in["wk"], fx_in["wv"])
    for k in ("Q", "K", "V", "S", "A", "O"):
        write_hex(os.path.join(d, f"expected_{k.lower()}.hex"), fx[k])
    fl = attention_float(M["x"], M["wq"], M["wk"], M["wv"])
    np.savetxt(os.path.join(d, "expected_float.txt"), fl["O"], fmt="%.6f")

    err = np.abs(np.array(fx["O"]) / 256 - fl["O"])
    sat = sum(abs(v) >= 32767 for r in fx["Q"] + fx["K"] + fx["V"] + fx["S"] for v in r)
    print(f"{name:20s} N={N:2d} D={D:2d} DK={DK:2d} | fixed-vs-float: "
          f"max err {err.max():.4f} ({err.max()*256:.2f} LSB), mean {err.mean():.5f} | saturations: {sat}")


if __name__ == "__main__":
    build("hand2", HAND)
    for name, N, D, DK, seed, rx in RANDOM_TESTS:
        build(name, random_inputs(N, D, DK, seed, rx))
