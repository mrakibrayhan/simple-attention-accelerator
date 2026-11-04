"""
fixed_model.py - BIT-EXACT fixed-point model of the attention hardware.

Every function here mirrors what the Verilog does, integer for integer,
so the RTL output must match this model EXACTLY (0 LSB difference).
The float model (golden_float.py) is used only to measure approximation error.

Number formats
--------------
  data   (X, W, Q, K, V, S, O) : Q8.8  signed 16-bit   (1 LSB = 1/256)
  exp / weights (e, w)         : Q0.16 unsigned 16-bit (1 LSB = 1/65536)
  multiplier operands          : 18-bit signed (Cyclone IV DSP = 18x18)
  accumulator                  : 48-bit signed
"""
import math
import numpy as np

FRAC = 8                 # Q8.8 fraction bits
DATA_W = 16
LUT_BITS = 6             # exp2 LUT: 64 entries indexed by top 6 fraction bits
LOG2E_Q16 = round(math.log2(math.e) * (1 << 16))   # 1.4427 in Q?.16 = 94548
RECIP_NUM_SHIFT = 32     # reciprocal R = floor(2^32 / sum)


# ---------------- basic fixed-point helpers ----------------
def sat(v, bits=DATA_W):
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return max(lo, min(hi, v))


def rnd_shift(acc, sh):
    """Round-half-up arithmetic right shift (same as Verilog (acc + 2^(sh-1)) >>> sh)."""
    return (acc + (1 << (sh - 1))) >> sh


def to_fixed(x, frac=FRAC):
    """float -> Q8.8 integer (round to nearest, saturate)."""
    return sat(int(round(x * (1 << frac))))


def to_float(v, frac=FRAC):
    return v / (1 << frac)


def scale_shift(dk):
    """1/sqrt(dk) as a right shift. Requires dk = 4^n (4, 16, 64, ...)."""
    s = math.isqrt(dk)
    assert s * s == dk and (s & (s - 1)) == 0, "d_k must be a power of 4 (4, 16, 64...)"
    return s.bit_length() - 1


# ---------------- exp approximation ----------------
def exp2_lut():
    """LUT[idx] = 2^-(idx+0.5)/64 in Q0.16 (mid-point of each bin -> halves max error)."""
    n = 1 << LUT_BITS
    return [int(round((1 << 16) * 2.0 ** (-(i + 0.5) / n))) for i in range(n)]


LUT = exp2_lut()


def exp_unit(u):
    """
    e = 2^(-u/256) in Q0.16, with u >= 0 in Q8.8.
      ui  = integer part      -> right shift
      idx = top LUT_BITS of the fraction -> LUT lookup
    """
    ui = u >> FRAC
    idx = (u & 0xFF) >> (FRAC - LUT_BITS)
    return 0 if ui >= 16 else LUT[idx] >> ui


def softmax_row(s_row):
    """Bit-exact softmax of one row of scaled scores (Q8.8 ints). Returns (e, R, w)."""
    m = max(s_row)
    e = []
    for s in s_row:
        d = sat(s - m)                      # <= 0
        nd = -d                             # 0 .. 32768
        u = (nd * LOG2E_Q16) >> 16          # (-d)*log2(e) in Q8.8   (e^d = 2^(-u))
        e.append(exp_unit(u))
    total = sum(e)                          # >= LUT[0] because the max element gives d = 0
    R = (1 << RECIP_NUM_SHIFT) // total     # one division per row
    w = [min((ei * R) >> 16, 0xFFFF) for ei in e]
    return e, R, w


# ---------------- full attention ----------------
def attention_fixed(X, WQ, WK, WV):
    """All inputs are 2-D lists of Q8.8 ints. Returns dict of every intermediate."""
    N, D = len(X), len(X[0])
    DK = len(WQ[0])
    sh = scale_shift(DK)

    def proj(W):
        return [[sat(rnd_shift(sum(X[i][k] * W[k][j] for k in range(D)), FRAC))
                 for j in range(DK)] for i in range(N)]

    Q, K, V = proj(WQ), proj(WK), proj(WV)
    S = [[sat(rnd_shift(sum(Q[i][k] * K[j][k] for k in range(DK)), FRAC + sh))
          for j in range(N)] for i in range(N)]
    E, R, A = [], [], []
    for i in range(N):
        e, r, w = softmax_row(S[i])
        E.append(e); R.append(r); A.append(w)
    O = [[sat(rnd_shift(sum(A[i][k] * V[k][j] for k in range(N)), 16))
          for j in range(DK)] for i in range(N)]
    return dict(Q=Q, K=K, V=V, S=S, E=E, R=R, A=A, O=O)


def attention_float(X, WQ, WK, WV):
    X, WQ, WK, WV = map(np.asarray, (X, WQ, WK, WV))
    Q, K, V = X @ WQ, X @ WK, X @ WV
    S = Q @ K.T / math.sqrt(WK.shape[1])
    e = np.exp(S - S.max(axis=1, keepdims=True))
    A = e / e.sum(axis=1, keepdims=True)
    return dict(Q=Q, K=K, V=V, S=S, A=A, O=A @ V)


# ---------------- hex file I/O ($readmemh format) ----------------
def write_hex(path, mat, bits=16):
    mask = (1 << bits) - 1
    with open(path, "w") as f:
        for row in mat:
            for v in (row if isinstance(row, (list, tuple)) else [row]):
                f.write(f"{v & mask:0{bits // 4}x}\n")


def read_hex(path, signed=True, bits=16):
    out = []
    with open(path) as f:
        for line in f:
            line = line.split("//")[0].strip()
            if not line:
                continue
            v = int(line, 16)
            if signed and v >= 1 << (bits - 1):
                v -= 1 << bits
            out.append(v)
    return out


def read_txt_matrix(path):
    """Human-readable decimal input: one matrix row per line, space separated."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if line:
                rows.append([float(t) for t in line.split()])
    return rows


if __name__ == "__main__":
    # Quick self-demo on the Step-1 hand example
    Xf = [[1, 0, 1, 0], [0, 2, 0, 2]]
    WQf = [[.5, 0, .5, 0], [0, .5, 0, .5], [.5, 0, 0, .5], [0, .5, .5, 0]]
    WKf = [[0, .5, 0, .5], [.5, 0, .5, 0], [0, .5, .5, 0], [.5, 0, 0, .5]]
    WVf = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    q = lambda M: [[to_fixed(v) for v in r] for r in M]
    fx = attention_fixed(q(Xf), q(WQf), q(WKf), q(WVf))
    fl = attention_float(Xf, WQf, WKf, WVf)
    print("LOG2E_Q16 =", LOG2E_Q16, " LUT[0..3] =", LUT[:4])
    print("S (Q8.8 ints) =", fx["S"], "-> float", [[to_float(v) for v in r] for r in fx["S"]])
    print("weights fixed =", [[round(w / 65536, 4) for w in r] for r in fx["A"]])
    print("weights float =", np.round(fl["A"], 4).tolist())
    print("O fixed =", [[round(to_float(v), 4) for v in r] for r in fx["O"]])
    print("O float =", np.round(fl["O"], 4).tolist())
    err = np.abs(np.array(fx["O"]) / 256 - fl["O"]).max()
    print(f"max |O_fixed - O_float| = {err:.5f}  ({err * 256:.2f} LSB of Q8.8)")
