// tb_mac.v - unit test for mac.v and round_sat.v
//   * MAC: random dot products of random length vs. a 64-bit integer reference
//   * round_sat: random values vs. an independent real-number formula floor(x/2^sh + 0.5)
`timescale 1ns/1ps
module tb_mac;
    reg clk = 0, rst = 1, en = 0, clr = 0;
    reg  signed [17:0] a, b;
    wire signed [47:0] acc;
    always #5 clk = ~clk;

    mac #(.AW(18), .ACC_W(48)) u_mac (.clk(clk), .rst(rst), .en(en), .clr(clr), .a(a), .b(b), .acc(acc));

    reg  signed [47:0] rs_acc;
    reg         [4:0]  rs_sh;
    wire signed [15:0] rs_y;
    round_sat u_rs (.acc(rs_acc), .sh(rs_sh), .y(rs_y));

    integer t, k, len, errors = 0;
    reg signed [63:0] ref_sum;
    real r;
    integer expv;

    initial begin
        $dumpfile("tb_mac.vcd");
        $dumpvars(0, tb_mac);
        repeat (2) @(posedge clk);
        rst = 0;

        // ---------- MAC test ----------
        for (t = 0; t < 200; t = t + 1) begin
            len = 1 + ($urandom % 64);
            ref_sum = 0;
            for (k = 0; k < len; k = k + 1) begin
                @(negedge clk);
                a = $random; b = $random;              // full 18-bit range, incl. extremes
                if (t == 0) begin a = -18'sd131072; b = -18'sd131072; end   // worst-case product
                en = 1; clr = (k == 0);
                ref_sum = ref_sum + a * b;
            end
            @(negedge clk); en = 0; clr = 0;
            if (acc !== ref_sum[47:0]) begin
                errors = errors + 1;
                $display("MAC FAIL t=%0d len=%0d got %0d exp %0d", t, len, acc, ref_sum);
            end
        end

        // ---------- round_sat test ----------
        for (t = 0; t < 5000; t = t + 1) begin
            rs_acc = {$random, $random};                       // random 64 -> keep 40 bits
            rs_acc = {{8{rs_acc[39]}}, rs_acc[39:0]};          // signed value, |x| < 2^39
            rs_sh  = 1 + ($urandom % 24);
            #1;
            r = rs_acc;                                        // exact (< 2^53)
            r = $floor(r / (2.0 ** rs_sh) + 0.5);
            if (r > 32767.0) r = 32767.0;
            if (r < -32768.0) r = -32768.0;
            expv = $rtoi(r);
            if (rs_y !== expv[15:0]) begin
                errors = errors + 1;
                if (errors < 10) $display("ROUND FAIL acc=%0d sh=%0d got %0d exp %0d", rs_acc, rs_sh, rs_y, expv);
            end
        end

        if (errors == 0) $display("tb_mac: PASS (200 dot products, 5000 round/saturate cases)");
        else             $display("tb_mac: FAIL (%0d errors)", errors);
        $finish;
    end
endmodule
