// ocloc-options: -cl-intel-256-GRF-per-thread
// C[M,N] (f32) = A[M,K] (f16) . W[N,K]^T for the GGUF K-quants with one scale per 16 values and no
// min: Q6_K (210-byte superblocks of 256) and Q3_K (110), DPAS on Xe2.
//
// The tiling of gemm_kquant.cl (a 64x32 C tile per subgroup, lane l owns W rows n0+l and
// n0+16+l, 8 subgroups as 4 (M) x 2 (N): 256x64 per workgroup). These superblocks are 2-byte
// multiples, so every other one starts 2 bytes off a dword: the kernels walk superblocks in PAIRS
// (K % 512 == 0, which keeps every row dword-aligned) and the odd one's loads are dword vector
// loads funnel-shifted by 2 bytes (`ld16`) — a compile-time choice per pair half. Values are
// assembled as unsigned bytes (0..63 / 0..7) with uint SWAR ops, and `deq16_*` (vISA) widens them
// into DPAS B rows, subtracting 1024 + the format's offset exactly (`0x6400 | q` is the half
// 1024 + q) and scaling by the sub-block scale.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define SG_M 64
#define SG_N 32
#define WG_M 256
#define WG_N 64
#define QK 256

// out row r = half2(q[2r] - OFF, q[2r+1] - OFF) * s, `q` 16 value bytes as uint4 (byte e of dword
// d = value 4d+e), `s2` the lane's scale twice (half2). 32-wide ops need M1_NM in a SIMD16 kernel;
// intermediates live in asm-local registers because IGC's inline asm has no early-clobber
// constraint.
#define DEQ16(NAME, NEG_BIAS)                                                                     \
    inline int8 NAME(uint4 q, uint s2) {                                                       \
        uint8 out;                                                                             \
        __asm__("{\n"                                                                          \
                ".decl XB v_type=G type=ub num_elts=256 align=GRF alias=<%1, 0>\n"             \
                ".decl OH v_type=G type=hf num_elts=256 align=GRF alias=<%0, 0>\n"             \
                ".decl TW v_type=G type=uw num_elts=256 align=GRF\n"                           \
                ".decl TH v_type=G type=hf num_elts=256 align=GRF alias=<TW, 0>\n"             \
                ".decl SD v_type=G type=ud num_elts=16 align=GRF\n"                            \
                ".decl SH v_type=G type=hf num_elts=32 align=GRF alias=<SD, 0>\n"              \
                "mov (M1_NM, 16) SD(0,0)<1> %2(0,0)<1;1,0>\n"                                  \
                "mov (M1_NM, 32) TW(0,0)<1> XB(0,0)<4;2,1>\n"                                  \
                "mov (M1_NM, 32) TW(1,0)<1> XB(0,2)<4;2,1>\n"                                  \
                "mov (M1_NM, 32) TW(2,0)<1> XB(1,0)<4;2,1>\n"                                  \
                "mov (M1_NM, 32) TW(3,0)<1> XB(1,2)<4;2,1>\n"                                  \
                "mov (M1_NM, 32) TW(4,0)<1> XB(2,0)<4;2,1>\n"                                  \
                "mov (M1_NM, 32) TW(5,0)<1> XB(2,2)<4;2,1>\n"                                  \
                "mov (M1_NM, 32) TW(6,0)<1> XB(3,0)<4;2,1>\n"                                  \
                "mov (M1_NM, 32) TW(7,0)<1> XB(3,2)<4;2,1>\n"                                  \
                "or (M1_NM, 32) TW(0,0)<1> TW(0,0)<1;1,0> 0x6400:uw\n"                         \
                "or (M1_NM, 32) TW(1,0)<1> TW(1,0)<1;1,0> 0x6400:uw\n"                         \
                "or (M1_NM, 32) TW(2,0)<1> TW(2,0)<1;1,0> 0x6400:uw\n"                         \
                "or (M1_NM, 32) TW(3,0)<1> TW(3,0)<1;1,0> 0x6400:uw\n"                         \
                "or (M1_NM, 32) TW(4,0)<1> TW(4,0)<1;1,0> 0x6400:uw\n"                         \
                "or (M1_NM, 32) TW(5,0)<1> TW(5,0)<1;1,0> 0x6400:uw\n"                         \
                "or (M1_NM, 32) TW(6,0)<1> TW(6,0)<1;1,0> 0x6400:uw\n"                         \
                "or (M1_NM, 32) TW(7,0)<1> TW(7,0)<1;1,0> 0x6400:uw\n"                         \
                "add (M1_NM, 32) TH(0,0)<1> TH(0,0)<1;1,0> " NEG_BIAS ":hf\n"                  \
                "add (M1_NM, 32) TH(1,0)<1> TH(1,0)<1;1,0> " NEG_BIAS ":hf\n"                  \
                "add (M1_NM, 32) TH(2,0)<1> TH(2,0)<1;1,0> " NEG_BIAS ":hf\n"                  \
                "add (M1_NM, 32) TH(3,0)<1> TH(3,0)<1;1,0> " NEG_BIAS ":hf\n"                  \
                "add (M1_NM, 32) TH(4,0)<1> TH(4,0)<1;1,0> " NEG_BIAS ":hf\n"                  \
                "add (M1_NM, 32) TH(5,0)<1> TH(5,0)<1;1,0> " NEG_BIAS ":hf\n"                  \
                "add (M1_NM, 32) TH(6,0)<1> TH(6,0)<1;1,0> " NEG_BIAS ":hf\n"                  \
                "add (M1_NM, 32) TH(7,0)<1> TH(7,0)<1;1,0> " NEG_BIAS ":hf\n"                  \
                "mul (M1_NM, 32) OH(0,0)<1> TH(0,0)<1;1,0> SH(0,0)<1;1,0>\n"                   \
                "mul (M1_NM, 32) OH(1,0)<1> TH(1,0)<1;1,0> SH(0,0)<1;1,0>\n"                   \
                "mul (M1_NM, 32) OH(2,0)<1> TH(2,0)<1;1,0> SH(0,0)<1;1,0>\n"                   \
                "mul (M1_NM, 32) OH(3,0)<1> TH(3,0)<1;1,0> SH(0,0)<1;1,0>\n"                   \
                "mul (M1_NM, 32) OH(4,0)<1> TH(4,0)<1;1,0> SH(0,0)<1;1,0>\n"                   \
                "mul (M1_NM, 32) OH(5,0)<1> TH(5,0)<1;1,0> SH(0,0)<1;1,0>\n"                   \
                "mul (M1_NM, 32) OH(6,0)<1> TH(6,0)<1;1,0> SH(0,0)<1;1,0>\n"                   \
                "mul (M1_NM, 32) OH(7,0)<1> TH(7,0)<1;1,0> SH(0,0)<1;1,0>\n"                   \
                "}\n"                                                                          \
                : "=rw"(out)                                                                   \
                : "rw"(q), "rw"(s2));                                                          \
        return as_int8(out);                                                                   \
    }

DEQ16(deq16_q6, "0xE420") // -(1024 + 32): Q6_K values are q - 32
DEQ16(deq16_q3, "0xE404") // -(1024 + 4):  Q3_K values are q - 4

// 16 bytes at `p`; `odd`: p sits 2 bytes past a dword boundary (the odd superblock of a pair).
inline uint4 ld16(__global const uchar* p, const int odd) {
    if (!odd)
        return vload4(0, (__global const uint*)p);
    __global const uint* q = (__global const uint*)(p - 2);
    const uint4 a = vload4(0, q);
    const uint b = q[4];
    return (uint4)((a.s0 >> 16) | (a.s1 << 16), (a.s1 >> 16) | (a.s2 << 16),
                   (a.s2 >> 16) | (a.s3 << 16), (a.s3 >> 16) | (b << 16));
}

// The superblock's f16 scale `d` at byte offset `off` (`odd` as for `ld16`).
inline half ld_d(__global const uchar* p, const int odd) {
    const uint w = *(__global const uint*)(p - 2 * odd);
    return as_half((ushort)(odd ? (w >> 16) : w));
}

// Accumulate one k16 block: A rows m0..m0+63 at column k, B operands b0/b1 (rows j = 0, 1).
#define DPAS_K16(k, b0, b1)                                                                      \
    {                                                                                            \
        ushort a0[32], a1[32];                                                                   \
        intel_sub_group_2d_block_read_16b_32r16x1c((__global void*)A, K * 2, M, K * 2,           \
                                                   (int2)(k, m0), a0);                           \
        intel_sub_group_2d_block_read_16b_32r16x1c((__global void*)A, K * 2, M, K * 2,           \
                                                   (int2)(k, m0 + 32), a1);                      \
        for (int i = 0; i < 4; i++) {                                                            \
            short8 ai = as_short8(vload8(0, &a0[i * 8]));                                        \
            acc[i][0] = intel_sub_group_f16_f16_matrix_mad_k16(ai, b0, acc[i][0]);               \
            acc[i][1] = intel_sub_group_f16_f16_matrix_mad_k16(ai, b1, acc[i][1]);               \
        }                                                                                        \
        for (int i = 0; i < 4; i++) {                                                            \
            short8 ai = as_short8(vload8(0, &a1[i * 8]));                                        \
            acc[4 + i][0] = intel_sub_group_f16_f16_matrix_mad_k16(ai, b0, acc[4 + i][0]);       \
            acc[4 + i][1] = intel_sub_group_f16_f16_matrix_mad_k16(ai, b1, acc[4 + i][1]);       \
        }                                                                                        \
    }

#define PROLOGUE(BLK)                                                                            \
    const int sg = get_sub_group_id();                                                           \
    const int lane = get_sub_group_local_id();                                                   \
    const int m0 = get_group_id(0) * WG_M + (sg >> 1) * SG_M;                                    \
    const int n0 = get_group_id(1) * WG_N + (sg & 1) * SG_N;                                     \
    const int nsb = K / QK;                                                                      \
    const size_t rowb = (size_t)nsb * BLK;                                                       \
    __global const uchar* row[2];                                                                \
    row[0] = W + (size_t)(n0 + lane) * rowb;                                                     \
    row[1] = row[0] + 16 * rowb;                                                                 \
    float8 acc[8][2];                                                                            \
    for (int i = 0; i < 8; i++) {                                                                \
        acc[i][0] = 0.0f;                                                                        \
        acc[i][1] = 0.0f;                                                                        \
    }

#define EPILOGUE                                                                                 \
    for (int i = 0; i < 8; i++) {                                                                \
        for (int j = 0; j < 2; j++) {                                                            \
            intel_sub_group_2d_block_write_32b_8r16x1c(                                          \
                (__global void*)C, N * 4, M, N * 4, (int2)(n0 + j * 16, m0 + i * 8),             \
                (__private uint*)&acc[i][j]);                                                    \
        }                                                                                        \
    }

// Q6_K superblock `s` (`odd` = s & 1): ql[128] (low 4 bits), qh[64] (high 2 bits), scales[16]
// (int8, one per 16 values), d. In half n (128 values), value 32g + l (g = 0..3, l = 0..31): low
// bits are nibble g/2 of ql[64n + 32(g%2) + l], high bits are bits 2g..2g+1 of qh[32n + l]; the
// scale is scales[8n + 2g + l/16].
#define Q6K_SB(s, odd)                                                                           \
    {                                                                                            \
        half sc[2][16];                                                                          \
        for (int j = 0; j < 2; j++) {                                                            \
            __global const uchar* blk = row[j] + (s) * 210;                                      \
            const half d = ld_d(blk + 208, odd);                                                 \
            const half16 v = convert_half16(as_char16(ld16(blk + 192, odd))) * d;                \
            for (int e = 0; e < 16; e++)                                                         \
                sc[j][e] = v[e];                                                                 \
        }                                                                                        \
        __attribute__((opencl_unroll_hint)) for (int n = 0; n < 2; n++) {                        \
            __attribute__((opencl_unroll_hint)) for (int l16 = 0; l16 < 2; l16++) {              \
                uint4 qh[2];                                                                     \
                for (int j = 0; j < 2; j++)                                                      \
                    qh[j] = ld16(row[j] + (s) * 210 + 128 + 32 * n + 16 * l16, odd);             \
                __attribute__((opencl_unroll_hint)) for (int g2 = 0; g2 < 2; g2++) {             \
                    uint4 ql[2];                                                                 \
                    for (int j = 0; j < 2; j++)                                                  \
                        ql[j] = ld16(row[j] + (s) * 210 + 64 * n + 32 * g2 + 16 * l16, odd);     \
                    __attribute__((opencl_unroll_hint)) for (int hi = 0; hi < 2; hi++) {         \
                        const int g = g2 + 2 * hi;                                               \
                        const int kb = 8 * n + 2 * g + l16;                                      \
                        int8 b[2];                                                               \
                        for (int j = 0; j < 2; j++) {                                            \
                            uint4 lo4 = hi ? ((ql[j] >> 4) & 0x0F0F0F0Fu) : (ql[j] & 0x0F0F0F0Fu); \
                            uint4 h2 = ((qh[j] >> (uint)(2 * g)) & 0x03030303u) << 4;            \
                            b[j] = deq16_q6(lo4 | h2, as_uint((half2)(sc[j][kb])));              \
                        }                                                                        \
                        DPAS_K16((s) * QK + 128 * n + 32 * g + 16 * l16, b[0], b[1])             \
                    }                                                                            \
                }                                                                                \
            }                                                                                    \
        }                                                                                        \
    }

// K % 512 == 0 (superblock pairs).
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void gemm_q6k(__global const half* A, __global const uchar* W, __global float* C, int M,
                       int N, int K) {
    PROLOGUE(210)
    for (int s = 0; s < nsb; s += 2) {
        Q6K_SB(s, 0)
        Q6K_SB(s + 1, 1)
    }
    EPILOGUE
}

// Q3_K superblock `s` (`odd` = s & 1): hmask[32] (high bit), qs[64] (low 2 bits), scales[12] (16
// 6-bit scales), d. In half n (128 values), value 32j + 16h + l (j = 0..3, h = 0..1, l = 0..15):
// low bits are bits 2j..2j+1 of qs[32n + 16h + l], the high bit is bit 4n+j of hmask[16h + l];
// scale index 8n + 2j + h. Value = (low | high << 2) - 4.
#define Q3K_SB(s, odd)                                                                           \
    {                                                                                            \
        half sc[2][16];                                                                          \
        uint4 hm[2][2];                                                                          \
        for (int j = 0; j < 2; j++) {                                                            \
            __global const uchar* blk = row[j] + (s) * 110;                                      \
            const half d = ld_d(blk + 108, odd);                                                 \
            const uint4 u = ld16(blk + 96, odd); /* 12 scale bytes (+4 of d / next) */           \
            const uint t = u.s2;                                                                 \
            const uint s0 = (u.s0 & 0x0F0F0F0Fu) | (((t >> 0) & 0x03030303u) << 4);             \
            const uint s1 = (u.s1 & 0x0F0F0F0Fu) | (((t >> 2) & 0x03030303u) << 4);             \
            const uint s2v = ((u.s0 >> 4) & 0x0F0F0F0Fu) | (((t >> 4) & 0x03030303u) << 4);      \
            const uint s3 = ((u.s1 >> 4) & 0x0F0F0F0Fu) | (((t >> 6) & 0x03030303u) << 4);      \
            const half16 v =                                                                     \
                (convert_half16(as_uchar16((uint4)(s0, s1, s2v, s3))) - (half)32.0h) * d;        \
            for (int e = 0; e < 16; e++)                                                         \
                sc[j][e] = v[e];                                                                 \
            for (int h = 0; h < 2; h++)                                                          \
                hm[j][h] = ld16(blk + 16 * h, odd);                                              \
        }                                                                                        \
        __attribute__((opencl_unroll_hint)) for (int n = 0; n < 2; n++) {                        \
            __attribute__((opencl_unroll_hint)) for (int h = 0; h < 2; h++) {                    \
                uint4 q[2];                                                                      \
                for (int j = 0; j < 2; j++)                                                      \
                    q[j] = ld16(row[j] + (s) * 110 + 32 + 32 * n + 16 * h, odd);                 \
                __attribute__((opencl_unroll_hint)) for (int jj = 0; jj < 4; jj++) {             \
                    const int kb = 8 * n + 2 * jj + h;                                           \
                    int8 b[2];                                                                   \
                    for (int j = 0; j < 2; j++) {                                                \
                        uint4 lo2 = (q[j] >> (uint)(2 * jj)) & 0x03030303u;                      \
                        uint4 hb = ((hm[j][h] >> (uint)(4 * n + jj)) & 0x01010101u) << 2;        \
                        b[j] = deq16_q3(lo2 | hb, as_uint((half2)(sc[j][kb])));                  \
                    }                                                                            \
                    DPAS_K16((s) * QK + 128 * n + 32 * jj + 16 * h, b[0], b[1])                  \
                }                                                                                \
            }                                                                                    \
        }                                                                                        \
    }

// K % 512 == 0 (superblock pairs).
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void gemm_q3k(__global const half* A, __global const uchar* W, __global float* C, int M,
                       int N, int K) {
    PROLOGUE(110)
    for (int s = 0; s < nsb; s += 2) {
        Q3K_SB(s, 0)
        Q3K_SB(s + 1, 1)
    }
    EPILOGUE
}

// ── Skinny GEMMs: M <= 16 rows (see gemm_kquant.cl's `gemm_kq_sk`) ─────────────────────────────
// The superblock bodies above are reused unchanged: only the A tile (one DPAS row block of 8, or
// two for 16 rows), the accumulators, the grid (256 columns per workgroup, K split over grid y in
// superblock PAIRS) and the partial-C write change. `MT` is the row-block count.
#undef DPAS_K16
#define DPAS_K16(k, b0, b1)                                                                      \
    {                                                                                            \
        short8 a_[2];                                                                            \
        if (MT == 1) {                                                                           \
            intel_sub_group_2d_block_read_16b_8r16x1c((__global void*)A, K * 2, M, K * 2,        \
                                                      (int2)(k, 0), (__private ushort*)&a_[0]);  \
        } else {                                                                                 \
            ushort a16_[16];                                                                     \
            intel_sub_group_2d_block_read_16b_16r16x1c((__global void*)A, K * 2, M, K * 2,       \
                                                       (int2)(k, 0), a16_);                      \
            a_[0] = as_short8(vload8(0, a16_));                                                  \
            a_[1] = as_short8(vload8(1, a16_));                                                  \
        }                                                                                        \
        for (int i = 0; i < MT; i++) {                                                           \
            acc[i][0] = intel_sub_group_f16_f16_matrix_mad_k16(a_[i], b0, acc[i][0]);            \
            acc[i][1] = intel_sub_group_f16_f16_matrix_mad_k16(a_[i], b1, acc[i][1]);            \
        }                                                                                        \
    }

#undef PROLOGUE
#define PROLOGUE(BLK)                                                                            \
    const int sg = get_sub_group_id();                                                           \
    const int lane = get_sub_group_local_id();                                                   \
    const int n0 = get_group_id(0) * 256 + sg * 32;                                              \
    if (n0 >= N)                                                                                 \
        return;                                                                                  \
    const int nsb = K / QK;                                                                      \
    const int per = (((nsb + nsplit - 1) / nsplit) + 1) & ~1;                                    \
    const int s0 = get_group_id(1) * per;                                                        \
    const int s1 = min(nsb, s0 + per);                                                           \
    const size_t rowb = (size_t)nsb * BLK;                                                       \
    __global const uchar* row[2];                                                                \
    row[0] = W + (size_t)min(n0 + lane, N - 1) * rowb;                                           \
    row[1] = W + (size_t)min(n0 + 16 + lane, N - 1) * rowb;                                      \
    float8 acc[2][2];                                                                            \
    for (int i = 0; i < 2; i++) {                                                                \
        acc[i][0] = 0.0f;                                                                        \
        acc[i][1] = 0.0f;                                                                        \
    }

#undef EPILOGUE
#define EPILOGUE                                                                                 \
    __global float* Cs = C + (size_t)get_group_id(1) * M * N;                                    \
    for (int i = 0; i < MT; i++) {                                                               \
        for (int j = 0; j < 2; j++) {                                                            \
            intel_sub_group_2d_block_write_32b_8r16x1c((__global void*)Cs, N * 4, M, N * 4,      \
                                                       (int2)(n0 + j * 16, i * 8),               \
                                                       (__private uint*)&acc[i][j]);             \
        }                                                                                        \
    }

#define KQ16_SKINNY(name, BLKB, SB, MT_)                                                         \
    __attribute__((intel_reqd_sub_group_size(16)))                                               \
    __attribute__((reqd_work_group_size(128, 1, 1)))                                             \
    __kernel void name(__global const half* A, __global const uchar* W, __global float* C,     \
                       int M, int N, int K, int nsplit) {                                        \
        const int MT = MT_;                                                                      \
        PROLOGUE(BLKB)                                                                           \
        for (int s = s0; s < s1; s += 2) {                                                       \
            SB(s, 0)                                                                             \
            SB(s + 1, 1)                                                                         \
        }                                                                                        \
        EPILOGUE                                                                                 \
    }
KQ16_SKINNY(gemm_q6k_m8, 210, Q6K_SB, 1)
KQ16_SKINNY(gemm_q6k_m16, 210, Q6K_SB, 2)
KQ16_SKINNY(gemm_q3k_m8, 110, Q3K_SB, 1)
KQ16_SKINNY(gemm_q3k_m16, 110, Q3K_SB, 2)
