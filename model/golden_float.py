"""
Step 1 - Floating-point golden model for scaled dot-product attention.
Prints every intermediate result so you can check your hand calculation.

    Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V

Run:  python3 golden_float.py
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)

# ---- Hand example: 2 tokens, d_model = 4, d_k = d_v = 4 -------------------
X = np.array([[1, 0, 1, 0],
              [0, 2, 0, 2]], dtype=float)          # 2 tokens x 4 features

W_Q = np.array([[1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 0, 1],
                [0, 1, 1, 0]], dtype=float) * 0.5

W_K = np.array([[0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 1, 0],
                [1, 0, 0, 1]], dtype=float) * 0.5

W_V = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]], dtype=float)        # identity -> V = X (easy to check)


def attention(X, W_Q, W_K, W_V, verbose=True):
    d_k = W_K.shape[1]
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    S = Q @ K.T
    S_scaled = S / np.sqrt(d_k)
    # numerically stable softmax: subtract the row max (the hardware will do this too)
    e = np.exp(S_scaled - S_scaled.max(axis=1, keepdims=True))
    A = e / e.sum(axis=1, keepdims=True)
    O = A @ V
    if verbose:
        for name, M in [("Q = X.W_Q", Q), ("K = X.W_K", K), ("V = X.W_V", V),
                        ("S = Q.K^T", S), (f"S / sqrt({d_k})", S_scaled),
                        ("softmax(S) (rows sum to 1)", A), ("O = softmax(S).V", O)]:
            print(f"\n{name}:\n{M}")
        print("\nrow sums of softmax:", A.sum(axis=1))
    return O


if __name__ == "__main__":
    attention(X, W_Q, W_K, W_V)
