// round_sat.v - accumulator -> 16-bit result:  sat16( (acc + 2^(sh-1)) >>> sh )
//
// Round-half-up (removes the -1/2 LSB bias of plain truncation), then
// saturate to [-32768, 32767] instead of wrapping around.
// sh is 8 (projections, Q8.8*Q8.8), 8+log2(sqrt(dk)) (scores, includes the 1/sqrt(dk) scale)
// or 16 (output, Q0.16 weight * Q8.8 value).
`timescale 1ns/1ps
module round_sat #(
    parameter ACC_W = 48
)(
    input  wire signed [ACC_W-1:0] acc,
    input  wire        [4:0]       sh,     // 1..31
    output reg  signed [15:0]      y
);
    reg signed [ACC_W-1:0] half, tmp, shifted;
    always @(*) begin
        half    = {{(ACC_W-1){1'b0}}, 1'b1} <<< (sh - 5'd1);
        tmp     = acc + half;
        shifted = tmp >>> sh;
        if      (shifted >  $signed(48'sd32767))  y = 16'sh7FFF;
        else if (shifted < -$signed(48'sd32768))  y = 16'sh8000;
        else                                      y = shifted[15:0];
    end
endmodule
