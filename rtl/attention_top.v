// attention_top.v - Scaled dot-product attention accelerator (sequential, single MAC)
//
//      O = softmax( (X*WQ) * (X*WK)^T / sqrt(DK) ) * (X*WV)
//
// Architecture
//   * ONE 18x18 MAC (mac.v) shared by every multiply, driven by a control FSM.
//   * Phase 1  : Q, K, V projections -> Q/K/V buffers           (3*N*DK dot products of length D)
//   * Phase 2  : row by row (row-streaming, so only ONE score row is stored):
//                  2a scores  s_j = Q_i . K_j  >>> (8 + log2 sqrt(DK))   + running row max
//                  2b exp     e_j = 2^(-(max - s_j)*log2e)              + running sum
//                  2c recip   R   = floor(2^32 / sum)                    (recip_div, 34 cycles)
//                  2d norm    w_j = e_j * R >> 16                        (Q0.16 weights)
//                  2e output  O_ij = sum_k w_k * V_kj  >>> 16
//   * Number formats: data Q8.8 signed 16-bit, weights Q0.16 unsigned, 48-bit accumulator.
//
// Interface
//   load : ld_en/ld_sel/ld_addr/ld_data write X (sel 0), WQ (1), WK (2), WV (3), row-major
//   run  : pulse start, wait for done
//   read : rd_addr -> rd_data returns O[rd_addr] (row-major, Q8.8)
//
// Requirement: DK must be a power of 4 so 1/sqrt(DK) is a pure shift (4, 16, 64, ...).
`timescale 1ns/1ps
module attention_top #(
    parameter N  = 4,     // tokens
    parameter D  = 8,     // model (embedding) dimension
    parameter DK = 4,     // head dimension (d_k = d_v)
    parameter AW = 16     // address width of the load / read ports
)(
    input  wire                 clk,
    input  wire                 rst,
    // load port
    input  wire                 ld_en,
    input  wire [1:0]           ld_sel,
    input  wire [AW-1:0]        ld_addr,
    input  wire signed [15:0]   ld_data,
    // control
    input  wire                 start,
    output reg                  busy,
    output reg                  done,
    output reg  [31:0]          cycles,
    // output read port
    input  wire [AW-1:0]        rd_addr,
    output wire signed [15:0]   rd_data
);
    // ------------------------------------------------------------------
    // Constants
    // ------------------------------------------------------------------
    localparam integer SCALE_SH = $clog2(DK) / 2;      // log2(sqrt(DK))
    localparam [4:0]   SH_PROJ  = 5'd8;
    localparam [4:0]   SH_SCORE = 8 + SCALE_SH;
    localparam [4:0]   SH_OUT   = 5'd16;
    localparam signed [17:0] LOG2E = 18'sd94548;       // log2(e) in Q.16

    // FSM states
    localparam [3:0] S_IDLE      = 4'd0,
                     S_PROJ_MAC  = 4'd1,  S_PROJ_WR  = 4'd2,
                     S_SCORE_MAC = 4'd3,  S_SCORE_WR = 4'd4,
                     S_EXP_MUL   = 4'd5,  S_EXP_WR   = 4'd6,
                     S_DIV_START = 4'd7,  S_DIV_WAIT = 4'd8,
                     S_NORM_MUL  = 4'd9,  S_NORM_WR  = 4'd10,
                     S_OUT_MAC   = 4'd11, S_OUT_WR   = 4'd12,
                     S_DONE      = 4'd13;

    // ------------------------------------------------------------------
    // Buffers (row-major)
    // ------------------------------------------------------------------
    reg signed [15:0] x_mem  [0:N*D-1];
    reg signed [15:0] wq_mem [0:D*DK-1];
    reg signed [15:0] wk_mem [0:D*DK-1];
    reg signed [15:0] wv_mem [0:D*DK-1];
    reg signed [15:0] q_mem  [0:N*DK-1];
    reg signed [15:0] k_mem  [0:N*DK-1];
    reg signed [15:0] v_mem  [0:N*DK-1];
    reg signed [15:0] s_row  [0:N-1];       // one row of scaled scores
    reg        [15:0] a_row  [0:N-1];       // one row of exp values, then weights (Q0.16)
    reg signed [15:0] o_mem  [0:N*DK-1];

    assign rd_data = o_mem[rd_addr];

    // load port
    always @(posedge clk) begin
        if (ld_en) begin
            case (ld_sel)
                2'd0: x_mem [ld_addr] <= ld_data;
                2'd1: wq_mem[ld_addr] <= ld_data;
                2'd2: wk_mem[ld_addr] <= ld_data;
                2'd3: wv_mem[ld_addr] <= ld_data;
            endcase
        end
    end

    // ------------------------------------------------------------------
    // Control state
    // ------------------------------------------------------------------
    reg [3:0]         state;
    reg [1:0]         mat;          // 0 = Q, 1 = K, 2 = V during projection
    reg [15:0]        i, j, k;      // row, column, reduction index
    reg signed [15:0] rowmax;
    reg [23:0]        sum;
    reg [17:0]        recip;

    // ------------------------------------------------------------------
    // Shared datapath
    // ------------------------------------------------------------------
    reg                mac_en, mac_clr;
    reg  signed [17:0] mac_a, mac_b;
    wire signed [47:0] acc;

    mac #(.AW(18), .ACC_W(48)) u_mac (
        .clk(clk), .rst(rst), .en(mac_en), .clr(mac_clr),
        .a(mac_a), .b(mac_b), .acc(acc));

    // rounding / saturation of the accumulator
    reg  [4:0]         rs_sh;
    wire signed [15:0] rs_y;
    round_sat #(.ACC_W(48)) u_rs (.acc(acc), .sh(rs_sh), .y(rs_y));

    // softmax helpers
    wire signed [16:0] d_raw  = s_row[j] - rowmax;                        // <= 0
    wire signed [15:0] d_sat  = (d_raw < -17'sd32768) ? -16'sd32768 : d_raw[15:0];
    wire signed [17:0] neg_d  = -{{2{d_sat[15]}}, d_sat};                 // 0 .. 32768
    wire        [15:0] exp_u  = acc[31:16];                               // (-d)*log2e, Q8.8
    wire        [15:0] exp_e;
    exp2_unit u_exp (.u(exp_u), .e(exp_e));
    wire        [15:0] w_val  = (|acc[47:32]) ? 16'hFFFF : acc[31:16];    // e*R >> 16, saturated

    // reciprocal divider
    wire        div_start = (state == S_DIV_START);
    wire        div_busy, div_done;
    wire [32:0] div_quot;
    recip_div #(.DW(24)) u_div (
        .clk(clk), .rst(rst), .start(div_start), .divisor(sum),
        .busy(div_busy), .done(div_done), .quot(div_quot));

    // selected projection weight
    reg signed [15:0] w_sel;
    always @(*) begin
        case (mat)
            2'd0:    w_sel = wq_mem[k*DK + j];
            2'd1:    w_sel = wk_mem[k*DK + j];
            default: w_sel = wv_mem[k*DK + j];
        endcase
    end

    // MAC operand multiplexer (the "Data Fetch" block of the reference diagram)
    always @(*) begin
        mac_en = 1'b0; mac_clr = 1'b0; mac_a = 18'sd0; mac_b = 18'sd0; rs_sh = SH_PROJ;
        case (state)
            S_PROJ_MAC: begin
                mac_en = 1'b1; mac_clr = (k == 0);
                mac_a  = {{2{x_mem[i*D + k][15]}}, x_mem[i*D + k]};
                mac_b  = {{2{w_sel[15]}}, w_sel};
            end
            S_SCORE_MAC: begin
                mac_en = 1'b1; mac_clr = (k == 0);
                mac_a  = {{2{q_mem[i*DK + k][15]}}, q_mem[i*DK + k]};
                mac_b  = {{2{k_mem[j*DK + k][15]}}, k_mem[j*DK + k]};
            end
            S_EXP_MUL: begin
                mac_en = 1'b1; mac_clr = 1'b1;
                mac_a  = neg_d;
                mac_b  = LOG2E;
            end
            S_NORM_MUL: begin
                mac_en = 1'b1; mac_clr = 1'b1;
                mac_a  = {2'b00, a_row[j]};
                mac_b  = recip;
            end
            S_OUT_MAC: begin
                mac_en = 1'b1; mac_clr = (k == 0);
                mac_a  = {2'b00, a_row[k]};
                mac_b  = {{2{v_mem[k*DK + j][15]}}, v_mem[k*DK + j]};
            end
            default: ;
        endcase
        case (state)
            S_SCORE_WR: rs_sh = SH_SCORE;
            S_OUT_WR:   rs_sh = SH_OUT;
            default:    rs_sh = SH_PROJ;
        endcase
    end

    // ------------------------------------------------------------------
    // Control FSM
    // ------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE; busy <= 1'b0; done <= 1'b0; cycles <= 32'd0;
            mat <= 0; i <= 0; j <= 0; k <= 0; rowmax <= 0; sum <= 0; recip <= 0;
        end else begin
            if (busy) cycles <= cycles + 32'd1;

            case (state)
            S_IDLE: if (start) begin
                busy <= 1'b1; done <= 1'b0; cycles <= 32'd0;
                mat <= 0; i <= 0; j <= 0; k <= 0;
                state <= S_PROJ_MAC;
            end

            // ---------------- Phase 1: Q, K, V = X * W ----------------
            S_PROJ_MAC: begin
                if (k == D-1) begin k <= 0; state <= S_PROJ_WR; end
                else            k <= k + 1;
            end
            S_PROJ_WR: begin
                case (mat)
                    2'd0:    q_mem[i*DK + j] <= rs_y;
                    2'd1:    k_mem[i*DK + j] <= rs_y;
                    default: v_mem[i*DK + j] <= rs_y;
                endcase
                state <= S_PROJ_MAC;
                if (j == DK-1) begin
                    j <= 0;
                    if (i == N-1) begin
                        i <= 0;
                        if (mat == 2'd2) state <= S_SCORE_MAC;    // all projections done
                        else             mat   <= mat + 1;
                    end else i <= i + 1;
                end else j <= j + 1;
            end

            // ---------------- Phase 2a: scores for row i ----------------
            S_SCORE_MAC: begin
                if (k == DK-1) begin k <= 0; state <= S_SCORE_WR; end
                else             k <= k + 1;
            end
            S_SCORE_WR: begin
                s_row[j] <= rs_y;
                if (j == 0 || rs_y > rowmax) rowmax <= rs_y;
                if (j == N-1) begin j <= 0; sum <= 0; state <= S_EXP_MUL; end
                else          begin j <= j + 1;       state <= S_SCORE_MAC; end
            end

            // ---------------- Phase 2b: exp + running sum ----------------
            S_EXP_MUL: state <= S_EXP_WR;
            S_EXP_WR: begin
                a_row[j] <= exp_e;
                sum      <= sum + {8'd0, exp_e};
                if (j == N-1) begin j <= 0; state <= S_DIV_START; end
                else          begin j <= j + 1; state <= S_EXP_MUL; end
            end

            // ---------------- Phase 2c: reciprocal of the sum ----------------
            S_DIV_START: state <= S_DIV_WAIT;
            S_DIV_WAIT: if (div_done) begin
                recip <= div_quot[17:0];
                state <= S_NORM_MUL;
            end

            // ---------------- Phase 2d: normalize ----------------
            S_NORM_MUL: state <= S_NORM_WR;
            S_NORM_WR: begin
                a_row[j] <= w_val;
                if (j == N-1) begin j <= 0; k <= 0; state <= S_OUT_MAC; end
                else          begin j <= j + 1;     state <= S_NORM_MUL; end
            end

            // ---------------- Phase 2e: output row i = w * V ----------------
            S_OUT_MAC: begin
                if (k == N-1) begin k <= 0; state <= S_OUT_WR; end
                else            k <= k + 1;
            end
            S_OUT_WR: begin
                o_mem[i*DK + j] <= rs_y;
                if (j == DK-1) begin
                    j <= 0;
                    if (i == N-1) state <= S_DONE;
                    else begin i <= i + 1; state <= S_SCORE_MAC; end
                end else begin
                    j <= j + 1; state <= S_OUT_MAC;
                end
            end

            S_DONE: begin
                busy  <= 1'b0;
                done  <= 1'b1;
                state <= S_IDLE;
            end
            default: state <= S_IDLE;
            endcase
        end
    end
endmodule
