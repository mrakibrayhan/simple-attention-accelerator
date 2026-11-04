// recip_div.v - sequential reciprocal:  quot = floor(2^32 / divisor)
//
// Restoring long division, 1 quotient bit per clock -> 33 cycles + 1.
// Used ONCE per softmax row (1/sum), so its latency is small compared with
// the N*dk MAC cycles of the row; a 1-bit-per-cycle divider is tiny
// (one subtractor + a few registers) versus a combinational divider.
//
// With divisor = sum of exp values (Q0.16, sum >= 0.9946 -> divisor >= 65182)
// the result is 1/sum in Q.16 and fits in 17 bits.
`timescale 1ns/1ps
module recip_div #(
    parameter DW = 24              // divisor width (sum of up to 256 exp values)
)(
    input  wire          clk,
    input  wire          rst,
    input  wire          start,
    input  wire [DW-1:0] divisor,
    output reg           busy,
    output reg           done,     // 1-cycle pulse, quot valid from then on
    output reg  [32:0]   quot
);
    reg [DW-1:0] rem;
    reg [DW-1:0] dvs;
    reg [5:0]    bitn;             // dividend bit being brought down: 32 .. 0

    // dividend = 2^32 -> only bit 32 is 1
    wire          in_bit = (bitn == 6'd32);
    wire [DW:0]   trial  = {rem, in_bit};
    wire          ge     = (trial >= {1'b0, dvs});
    wire [DW:0]   diff   = trial - {1'b0, dvs};

    always @(posedge clk) begin
        if (rst) begin
            busy <= 1'b0; done <= 1'b0; rem <= 0; quot <= 0; bitn <= 0; dvs <= 0;
        end else begin
            done <= 1'b0;
            if (start && !busy) begin
                busy <= 1'b1; rem <= 0; quot <= 0; bitn <= 6'd32; dvs <= divisor;
            end else if (busy) begin
                rem        <= ge ? diff[DW-1:0] : trial[DW-1:0];
                quot[bitn] <= ge;
                if (bitn == 0) begin busy <= 1'b0; done <= 1'b1; end
                else             bitn <= bitn - 6'd1;
            end
        end
    end
endmodule
