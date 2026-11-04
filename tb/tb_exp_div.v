// tb_exp_div.v - unit tests for exp2_unit.v and recip_div.v
//   * exp2_unit: EXHAUSTIVE over all 65536 inputs, checked against real 2^(-u/256)
//                with the analytic error bound of a 64-entry mid-point LUT (0.62 %) + 1 LSB
//   * recip_div: edge + random divisors vs. 64-bit integer division, and latency check
`timescale 1ns/1ps
module tb_exp_div;
    reg clk = 0, rst = 1;
    always #5 clk = ~clk;

    // ---------------- exp2_unit ----------------
    reg  [15:0] u;
    wire [15:0] e;
    exp2_unit u_exp (.u(u), .e(e));

    // ---------------- recip_div ----------------
    reg         start = 0;
    reg  [23:0] divisor;
    wire        busy, done;
    wire [32:0] quot;
    recip_div #(.DW(24)) u_div (.clk(clk), .rst(rst), .start(start), .divisor(divisor),
                                .busy(busy), .done(done), .quot(quot));

    integer i, errors = 0, cycles, max_rel_ppm = 0, rel_ppm;
    real    t, err, worst = 0.0;
    reg [63:0] expq;

    task run_div(input [23:0] d);
        begin
            @(negedge clk); divisor = d; start = 1;
            @(negedge clk); start = 0; cycles = 1;
            while (!done) begin @(negedge clk); cycles = cycles + 1; end
            expq = 64'h1_0000_0000 / d;
            if (quot !== expq[32:0]) begin
                errors = errors + 1;
                $display("DIV FAIL d=%0d got %0d exp %0d", d, quot, expq);
            end
        end
    endtask

    initial begin
        $dumpfile("tb_exp_div.vcd");
        $dumpvars(0, tb_exp_div);

        // ---- exp: exhaustive ----
        for (i = 0; i < 65536; i = i + 1) begin
            u = i; #1;
            t = 65536.0 * (2.0 ** (-i / 256.0));
            err = (e > t) ? e - t : t - e;
            if (err > 0.0063 * t + 1.0) begin
                errors = errors + 1;
                if (errors < 10) $display("EXP FAIL u=%0d got %0d true %f", i, e, t);
            end
            if (t > 1024.0 && err / t > worst) worst = err / t;   // rel. error where it matters
        end
        $display("exp2_unit: worst relative error (e > 1/64) = %0.3f %%", worst * 100.0);

        // ---- divider ----
        repeat (2) @(posedge clk); rst = 0;
        run_div(24'd65182);              // smallest possible sum (one element, d = 0)
        run_div(24'd65536);
        run_div(24'd16 * 65182);         // 16 equal elements
        run_div(24'hFFFFFF);             // largest divisor
        for (i = 0; i < 300; i = i + 1)
            run_div(24'd65182 + ($urandom % (24'hFFFFFF - 24'd65182)));
        $display("recip_div: latency = %0d cycles (start -> done)", cycles);

        if (errors == 0) $display("tb_exp_div: PASS (65536 exp inputs, 304 divisions)");
        else             $display("tb_exp_div: FAIL (%0d errors)", errors);
        $finish;
    end
endmodule
