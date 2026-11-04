| Test | N | D | d_k | Bit-exact vs model | Max error vs float | Mean error | Cycles | @100 MHz |
|---|---|---|---|---|---|---|---|---|
| hand2 | 2 | 4 | 4 | PASS | 0.0027 (0.70 LSB) | 0.00106 | 251 | 2.51 µs |
| rand_n4_d8_dk4 | 4 | 8 | 4 | PASS | 0.0033 (0.84 LSB) | 0.00085 | 797 | 7.97 µs |
| sharp_n8_d16_dk4 | 8 | 16 | 4 | PASS | 0.0145 (3.71 LSB) | 0.00342 | 2,777 | 27.77 µs |
| rand_n8_d16_dk16 | 8 | 16 | 16 | PASS | 0.0048 (1.23 LSB) | 0.00135 | 9,305 | 93.05 µs |
| rand_n16_d64_dk64 | 16 | 64 | 64 | PASS | 0.0058 (1.48 LSB) | 0.00127 | 235,313 | 2353.13 µs |
