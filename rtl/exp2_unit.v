// exp2_unit.v - e = 2^(-u) for u >= 0, combinational.
//
//   u  : unsigned Q8.8  (u = -(s - max) * log2(e), so e^(s-max) = 2^(-u))
//   e  : unsigned Q0.16 (0 .. 0.9946)
//
//   2^(-u) = 2^(-int(u)) * 2^(-frac(u))
//            \_ shift _/   \_ 64-entry LUT on the top 6 fraction bits _/
// Results below 2^-16 (int(u) >= 16) flush to 0.
`timescale 1ns/1ps
module exp2_unit (
    input  wire [15:0] u,
    output wire [15:0] e
);
    wire [7:0]  u_int = u[15:8];
    wire [5:0]  idx   = u[7:2];
    wire [15:0] lut_val;

    exp2_lut u_lut (.idx(idx), .data(lut_val));

    assign e = (u_int >= 8'd16) ? 16'd0 : (lut_val >> u_int[3:0]);
endmodule
