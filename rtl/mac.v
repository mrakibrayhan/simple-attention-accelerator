// mac.v - one 18x18 signed multiply-accumulate unit (the ONLY multiplier in the design)
//
//   if (en) acc <= (clr ? 0 : acc) + a*b
//
// clr=1 on the first term of a dot product starts a new sum in the same cycle
// (no dead cycle), and clr=1 alone gives a plain registered multiply.
// 18x18 matches one Cyclone IV embedded multiplier; the 36-bit product is
// sign-extended into a 48-bit accumulator so long dot products cannot overflow.
`timescale 1ns/1ps
module mac #(
    parameter AW    = 18,   // operand width
    parameter ACC_W = 48    // accumulator width
)(
    input  wire                    clk,
    input  wire                    rst,
    input  wire                    en,
    input  wire                    clr,
    input  wire signed [AW-1:0]    a,
    input  wire signed [AW-1:0]    b,
    output reg  signed [ACC_W-1:0] acc
);
    wire signed [2*AW-1:0]  prod     = a * b;
    wire signed [ACC_W-1:0] prod_ext = {{(ACC_W-2*AW){prod[2*AW-1]}}, prod};

    always @(posedge clk) begin
        if (rst)      acc <= {ACC_W{1'b0}};
        else if (en)  acc <= (clr ? {ACC_W{1'b0}} : acc) + prod_ext;
    end
endmodule
