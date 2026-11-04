// tb_attention.v - system testbench for attention_top
//
// Run it INSIDE a test folder (tests/<name>/). It reads from the current directory:
//     x.hex  wq.hex  wk.hex  wv.hex      ($readmemh, one 16-bit hex word per line, row-major)
// and writes:
//     output.hex        final O (hex, Q8.8)          <- compared bit-exactly by scripts/check.py
//     output.txt        final O (decimal, readable)
//     hw_q.hex hw_k.hex hw_v.hex hw_s.hex hw_a.hex   intermediate buffers (for debugging)
//     cycles.txt        latency in clock cycles
//     attention.vcd     waveform dump (skip with +novcd for very large tests)
//
// N, D, DK are set from the command line:  iverilog -P tb_attention.N=4 -P tb_attention.D=8 ...
`timescale 1ns/1ps
module tb_attention;
    parameter N  = 4;
    parameter D  = 8;
    parameter DK = 4;
    parameter TIMEOUT = 5_000_000;     // cycles

    reg clk = 0, rst = 1;
    always #5 clk = ~clk;              // 100 MHz

    // DUT ports
    reg               ld_en = 0;
    reg  [1:0]        ld_sel = 0;
    reg  [15:0]       ld_addr = 0;
    reg  signed [15:0] ld_data = 0;
    reg               start = 0;
    wire              busy, done;
    wire [31:0]       cycles;
    reg  [15:0]       rd_addr = 0;
    wire signed [15:0] rd_data;

    attention_top #(.N(N), .D(D), .DK(DK)) dut (
        .clk(clk), .rst(rst),
        .ld_en(ld_en), .ld_sel(ld_sel), .ld_addr(ld_addr), .ld_data(ld_data),
        .start(start), .busy(busy), .done(done), .cycles(cycles),
        .rd_addr(rd_addr), .rd_data(rd_data));

    // input images read from text files
    reg [15:0] x_f  [0:N*D-1];
    reg [15:0] wq_f [0:D*DK-1];
    reg [15:0] wk_f [0:D*DK-1];
    reg [15:0] wv_f [0:D*DK-1];

    integer n, f_out, f_txt, f_s, f_a, f_q, f_k, f_v, f_c, cnt;

    // stream one matrix into the DUT load port
    task load(input [1:0] sel, input integer len);
        integer t;
        begin
            for (t = 0; t < len; t = t + 1) begin
                @(negedge clk);
                ld_en = 1; ld_sel = sel; ld_addr = t;
                case (sel)
                    2'd0: ld_data = x_f[t];
                    2'd1: ld_data = wq_f[t];
                    2'd2: ld_data = wk_f[t];
                    2'd3: ld_data = wv_f[t];
                endcase
            end
            @(negedge clk); ld_en = 0;
        end
    endtask

    // log intermediate score / weight rows as the FSM produces them
    always @(posedge clk) begin
        if (dut.state == dut.S_SCORE_WR) $fwrite(f_s, "%04h\n", dut.rs_y[15:0]);
        if (dut.state == dut.S_NORM_WR)  $fwrite(f_a, "%04h\n", dut.w_val);
    end

    initial begin
        if (!$test$plusargs("novcd")) begin          // +novcd skips the dump (big tests)
            $dumpfile("attention.vcd");
            $dumpvars(0, tb_attention);
        end

        $readmemh("x.hex",  x_f);
        $readmemh("wq.hex", wq_f);
        $readmemh("wk.hex", wk_f);
        $readmemh("wv.hex", wv_f);
        f_s = $fopen("hw_s.hex", "w");
        f_a = $fopen("hw_a.hex", "w");

        $display("tb_attention: N=%0d D=%0d DK=%0d", N, D, DK);
        repeat (3) @(posedge clk);
        rst = 0;

        load(2'd0, N*D);
        load(2'd1, D*DK);
        load(2'd2, D*DK);
        load(2'd3, D*DK);

        @(negedge clk); start = 1;
        @(negedge clk); start = 0;

        cnt = 0;
        while (!done && cnt < TIMEOUT) begin @(posedge clk); cnt = cnt + 1; end
        if (!done) begin $display("ERROR: timeout after %0d cycles", cnt); $finish; end
        $display("tb_attention: done in %0d cycles", cycles);

        // ---- final output (via the read port) ----
        f_out = $fopen("output.hex", "w");
        f_txt = $fopen("output.txt", "w");
        for (n = 0; n < N*DK; n = n + 1) begin
            rd_addr = n; #1;
            $fwrite(f_out, "%04h\n", rd_data[15:0]);
            $fwrite(f_txt, "%9.5f%s", $itor(rd_data) / 256.0, ((n % DK) == DK-1) ? "\n" : " ");
        end
        $fclose(f_out); $fclose(f_txt);

        // ---- intermediate buffers (hierarchical access, debug only) ----
        f_q = $fopen("hw_q.hex", "w"); f_k = $fopen("hw_k.hex", "w"); f_v = $fopen("hw_v.hex", "w");
        for (n = 0; n < N*DK; n = n + 1) begin
            $fwrite(f_q, "%04h\n", dut.q_mem[n][15:0]);
            $fwrite(f_k, "%04h\n", dut.k_mem[n][15:0]);
            $fwrite(f_v, "%04h\n", dut.v_mem[n][15:0]);
        end
        $fclose(f_q); $fclose(f_k); $fclose(f_v); $fclose(f_s); $fclose(f_a);

        f_c = $fopen("cycles.txt", "w"); $fwrite(f_c, "%0d\n", cycles); $fclose(f_c);
        $display("tb_attention: wrote output.hex, output.txt, hw_*.hex, cycles.txt");
        $finish;
    end
endmodule
