// ocloc-options: -cl-intel-256-GRF-per-thread
// C[M,N] (f32) = A[M,K] (f16) . W[N,K]^T, W in GGUF IQ4_XS (136-byte superblocks of 256: d, 6-bit
// sub-block scales, 4-bit indices into the IQ4_NL codebook), DPAS on Xe2.
//
// The tiling of gemm_kquant.cl (a 64x32 C tile per subgroup, lane = W row of DPAS's B operand,
// 256x64 per workgroup). The codebook goes through a byte-pair table in shared memory,
// lut[a | b << 4] = half2(kv[a], kv[b]), so one gather gives a B row's k-pair; the sub-block
// scale is applied by `scale_b` (vISA, as in gemm_kquant.cl's `deq_b`). A sparse table indexed by
// the nibbles in place (lut[a | b << 8], offsets `(q << 2) & 0x3C3C`) needs fewer ALU ops but ran
// 4% slower — its entries 1 KB apart share SLM banks; the gathers, one per B row, are the limit.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define SG_M 64
#define SG_N 32
#define WG_M 256
#define WG_N 64
#define QK 256
#define BLK 136

__constant char KV[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                          1,    13,   25,  38,  53,  69,  89,  113};

// out row r = half2 row r of `v` times the lane's scale (`s2`: the scale twice, half2). The
// multiply runs on the packed k-pairs in place; OpenCL C would de-interleave them. 32-wide ops
// need M1_NM in a SIMD16 kernel; the scale is copied to an asm-local register first because IGC's
// inline asm has no early-clobber constraint (`out` may share registers with the inputs).
inline int8 scale_b(uint8 v, uint s2) {
    uint8 out;
    __asm__("{\n"
            ".decl VH v_type=G type=hf num_elts=256 align=GRF alias=<%1, 0>\n"
            ".decl OH v_type=G type=hf num_elts=256 align=GRF alias=<%0, 0>\n"
            ".decl SD v_type=G type=ud num_elts=16 align=GRF\n"
            ".decl SH v_type=G type=hf num_elts=32 align=GRF alias=<SD, 0>\n"
            ".decl TH v_type=G type=hf num_elts=256 align=GRF\n"
            "mov (M1_NM, 16) SD(0,0)<1> %2(0,0)<1;1,0>\n"
            "mul (M1_NM, 32) TH(0,0)<1> VH(0,0)<1;1,0> SH(0,0)<1;1,0>\n"
            "mul (M1_NM, 32) TH(1,0)<1> VH(1,0)<1;1,0> SH(0,0)<1;1,0>\n"
            "mul (M1_NM, 32) TH(2,0)<1> VH(2,0)<1;1,0> SH(0,0)<1;1,0>\n"
            "mul (M1_NM, 32) TH(3,0)<1> VH(3,0)<1;1,0> SH(0,0)<1;1,0>\n"
            "mul (M1_NM, 32) TH(4,0)<1> VH(4,0)<1;1,0> SH(0,0)<1;1,0>\n"
            "mul (M1_NM, 32) TH(5,0)<1> VH(5,0)<1;1,0> SH(0,0)<1;1,0>\n"
            "mul (M1_NM, 32) TH(6,0)<1> VH(6,0)<1;1,0> SH(0,0)<1;1,0>\n"
            "mul (M1_NM, 32) TH(7,0)<1> VH(7,0)<1;1,0> SH(0,0)<1;1,0>\n"
            "mov (M1_NM, 32) OH(0,0)<1> TH(0,0)<1;1,0>\n"
            "mov (M1_NM, 32) OH(1,0)<1> TH(1,0)<1;1,0>\n"
            "mov (M1_NM, 32) OH(2,0)<1> TH(2,0)<1;1,0>\n"
            "mov (M1_NM, 32) OH(3,0)<1> TH(3,0)<1;1,0>\n"
            "mov (M1_NM, 32) OH(4,0)<1> TH(4,0)<1;1,0>\n"
            "mov (M1_NM, 32) OH(5,0)<1> TH(5,0)<1;1,0>\n"
            "mov (M1_NM, 32) OH(6,0)<1> TH(6,0)<1;1,0>\n"
            "mov (M1_NM, 32) OH(7,0)<1> TH(7,0)<1;1,0>\n"
            "}\n"
            : "=rw"(out)
            : "rw"(v), "rw"(s2));
    return as_int8(out);
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void gemm_iq4xs(__global const half* A, __global const uchar* W, __global float* C,
                         int M, int N, int K) {
    __local uint lut[256];
    for (int i = get_local_id(0); i < 256; i += 128)
        lut[i] = as_uint((half2)((half)KV[i & 15], (half)KV[i >> 4]));
    barrier(CLK_LOCAL_MEM_FENCE);

    const int sg = get_sub_group_id();
    const int lane = get_sub_group_local_id();
    const int m0 = get_group_id(0) * WG_M + (sg >> 1) * SG_M;
    const int n0 = get_group_id(1) * WG_N + (sg & 1) * SG_N;
    const int nsb = K / QK;
    const size_t rowb = (size_t)nsb * BLK;
    __global const uchar* row0 = W + (size_t)(n0 + lane) * rowb;
    __global const uchar* row1 = row0 + 16 * rowb;

    float8 acc[8][2];
    for (int i = 0; i < 8; i++) {
        acc[i][0] = 0.0f;
        acc[i][1] = 0.0f;
    }

    for (int s = 0; s < nsb; s++) {
        __global const uchar* blk[2] = {row0 + s * BLK, row1 + s * BLK};
        uint2 hd[2];
        hd[0] = vload2(0, (__global const uint*)blk[0]);
        hd[1] = vload2(0, (__global const uint*)blk[1]);
        __attribute__((opencl_unroll_hint))
        for (int ib = 0; ib < 8; ib++) {
            // Sub-block ib: 16 bytes, low nibbles = values 0..15, high = 16..31.
            uint4 q[2];
            uint s2[2];
            for (int j = 0; j < 2; j++) {
                q[j] = vload4(0, (__global const uint*)(blk[j] + 8 + ib * 16));
                // d (low 16 bits of hd.x), scales_h (high 16), scales_l (hd.y)
                const int ls = ((hd[j].y >> (4 * ib)) & 0xF) | (((hd[j].x >> (16 + 2 * ib)) & 3) << 4);
                const half dl = as_half((ushort)hd[j].x) * (half)(ls - 32);
                s2[j] = as_uint((half2)(dl));
            }
            __attribute__((opencl_unroll_hint))
            for (int h = 0; h < 2; h++) {
                const int k = s * QK + ib * 32 + h * 16;
                ushort a0[32], a1[32];
                intel_sub_group_2d_block_read_16b_32r16x1c(
                    (__global void*)A, K * 2, M, K * 2, (int2)(k, m0), a0);
                intel_sub_group_2d_block_read_16b_32r16x1c(
                    (__global void*)A, K * 2, M, K * 2, (int2)(k, m0 + 32), a1);
                int8 b[2];
                for (int j = 0; j < 2; j++) {
                    uint4 x = h ? ((q[j] >> 4) & 0x0F0F0F0Fu) : (q[j] & 0x0F0F0F0Fu);
                    // row 2d: values 4d, 4d+1 (bytes 0, 1 of x.d); row 2d+1: bytes 2, 3
                    uint8 v;
                    v.s0 = lut[(x.s0 & 0xFu) | ((x.s0 >> 4) & 0xF0u)];
                    v.s1 = lut[((x.s0 >> 16) & 0xFu) | ((x.s0 >> 20) & 0xF0u)];
                    v.s2 = lut[(x.s1 & 0xFu) | ((x.s1 >> 4) & 0xF0u)];
                    v.s3 = lut[((x.s1 >> 16) & 0xFu) | ((x.s1 >> 20) & 0xF0u)];
                    v.s4 = lut[(x.s2 & 0xFu) | ((x.s2 >> 4) & 0xF0u)];
                    v.s5 = lut[((x.s2 >> 16) & 0xFu) | ((x.s2 >> 20) & 0xF0u)];
                    v.s6 = lut[(x.s3 & 0xFu) | ((x.s3 >> 4) & 0xF0u)];
                    v.s7 = lut[((x.s3 >> 16) & 0xFu) | ((x.s3 >> 20) & 0xF0u)];
                    b[j] = scale_b(v, s2[j]);
                }
                for (int i = 0; i < 4; i++) {
                    short8 ai = as_short8(vload8(0, &a0[i * 8]));
                    acc[i][0] = intel_sub_group_f16_f16_matrix_mad_k16(ai, b[0], acc[i][0]);
                    acc[i][1] = intel_sub_group_f16_f16_matrix_mad_k16(ai, b[1], acc[i][1]);
                }
                for (int i = 0; i < 4; i++) {
                    short8 ai = as_short8(vload8(0, &a1[i * 8]));
                    acc[4 + i][0] = intel_sub_group_f16_f16_matrix_mad_k16(ai, b[0], acc[4 + i][0]);
                    acc[4 + i][1] = intel_sub_group_f16_f16_matrix_mad_k16(ai, b[1], acc[4 + i][1]);
                }
            }
        }
    }

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 2; j++) {
            intel_sub_group_2d_block_write_32b_8r16x1c(
                (__global void*)C, N * 4, M, N * 4, (int2)(n0 + j * 16, m0 + i * 8),
                (__private uint*)&acc[i][j]);
        }
    }
}

// 16 bytes at `p`; `odd`: p sits 2 bytes past a dword boundary.
inline uint4 ld16(__global const uchar* p, const int odd) {
    if (!odd)
        return vload4(0, (__global const uint*)p);
    __global const uint* q = (__global const uint*)(p - 2);
    const uint4 a = vload4(0, q);
    const uint b = q[4];
    return (uint4)((a.s0 >> 16) | (a.s1 << 16), (a.s1 >> 16) | (a.s2 << 16),
                   (a.s2 >> 16) | (a.s3 << 16), (a.s3 >> 16) | (b << 16));
}

// An f16 at `p` (`odd` as for `ld16`).
inline half ld_d(__global const uchar* p, const int odd) {
    const uint w = *(__global const uint*)(p - 2 * odd);
    return as_half((ushort)(odd ? (w >> 16) : w));
}

// IQ4_NL: 18-byte blocks of 32 (d, then 16 bytes: low nibbles = values 0..15, high = 16..31),
// the same codebook as IQ4_XS. Blocks are walked in 36-byte pairs, which start dword-aligned when
// K % 64 == 0: the even block's qs and the odd block's d sit 2 bytes off a dword (`ld16`/`ld_d`).
#define IQ4NL_BLOCK(b, odd)                                                                      \
    {                                                                                            \
        uint4 q[2];                                                                              \
        uint s2[2];                                                                              \
        for (int j = 0; j < 2; j++) {                                                            \
            __global const uchar* blk = rows[j] + (size_t)(b) * 18;                              \
            s2[j] = as_uint((half2)(ld_d(blk, odd)));                                            \
            q[j] = ld16(blk + 2, !(odd));                                                        \
        }                                                                                        \
        __attribute__((opencl_unroll_hint)) for (int h = 0; h < 2; h++) {                        \
            const int k = (b) * 32 + h * 16;                                                     \
            ushort a0[32], a1[32];                                                               \
            intel_sub_group_2d_block_read_16b_32r16x1c((__global void*)A, K * 2, M, K * 2,       \
                                                       (int2)(k, m0), a0);                       \
            intel_sub_group_2d_block_read_16b_32r16x1c((__global void*)A, K * 2, M, K * 2,       \
                                                       (int2)(k, m0 + 32), a1);                  \
            int8 bb[2];                                                                          \
            for (int j = 0; j < 2; j++) {                                                        \
                uint4 x = h ? ((q[j] >> 4) & 0x0F0F0F0Fu) : (q[j] & 0x0F0F0F0Fu);                \
                uint8 v;                                                                         \
                v.s0 = lut[(x.s0 & 0xFu) | ((x.s0 >> 4) & 0xF0u)];                               \
                v.s1 = lut[((x.s0 >> 16) & 0xFu) | ((x.s0 >> 20) & 0xF0u)];                      \
                v.s2 = lut[(x.s1 & 0xFu) | ((x.s1 >> 4) & 0xF0u)];                               \
                v.s3 = lut[((x.s1 >> 16) & 0xFu) | ((x.s1 >> 20) & 0xF0u)];                      \
                v.s4 = lut[(x.s2 & 0xFu) | ((x.s2 >> 4) & 0xF0u)];                               \
                v.s5 = lut[((x.s2 >> 16) & 0xFu) | ((x.s2 >> 20) & 0xF0u)];                      \
                v.s6 = lut[(x.s3 & 0xFu) | ((x.s3 >> 4) & 0xF0u)];                               \
                v.s7 = lut[((x.s3 >> 16) & 0xFu) | ((x.s3 >> 20) & 0xF0u)];                      \
                bb[j] = scale_b(v, s2[j]);                                                       \
            }                                                                                    \
            for (int i = 0; i < 4; i++) {                                                        \
                short8 ai = as_short8(vload8(0, &a0[i * 8]));                                    \
                acc[i][0] = intel_sub_group_f16_f16_matrix_mad_k16(ai, bb[0], acc[i][0]);        \
                acc[i][1] = intel_sub_group_f16_f16_matrix_mad_k16(ai, bb[1], acc[i][1]);        \
            }                                                                                    \
            for (int i = 0; i < 4; i++) {                                                        \
                short8 ai = as_short8(vload8(0, &a1[i * 8]));                                    \
                acc[4 + i][0] = intel_sub_group_f16_f16_matrix_mad_k16(ai, bb[0], acc[4 + i][0]); \
                acc[4 + i][1] = intel_sub_group_f16_f16_matrix_mad_k16(ai, bb[1], acc[4 + i][1]); \
            }                                                                                    \
        }                                                                                        \
    }

// K % 64 == 0 (block pairs).
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void gemm_iq4nl(__global const half* A, __global const uchar* W, __global float* C,
                         int M, int N, int K) {
    __local uint lut[256];
    for (int i = get_local_id(0); i < 256; i += 128)
        lut[i] = as_uint((half2)((half)KV[i & 15], (half)KV[i >> 4]));
    barrier(CLK_LOCAL_MEM_FENCE);

    const int sg = get_sub_group_id();
    const int lane = get_sub_group_local_id();
    const int m0 = get_group_id(0) * WG_M + (sg >> 1) * SG_M;
    const int n0 = get_group_id(1) * WG_N + (sg & 1) * SG_N;
    const size_t rowb = (size_t)(K / 32) * 18;
    __global const uchar* rows[2];
    rows[0] = W + (size_t)(n0 + lane) * rowb;
    rows[1] = rows[0] + 16 * rowb;

    float8 acc[8][2];
    for (int i = 0; i < 8; i++) {
        acc[i][0] = 0.0f;
        acc[i][1] = 0.0f;
    }
    for (int bp = 0; bp < K / 64; bp++) {
        IQ4NL_BLOCK(2 * bp, 0)
        IQ4NL_BLOCK(2 * bp + 1, 1)
    }
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 2; j++) {
            intel_sub_group_2d_block_write_32b_8r16x1c(
                (__global void*)C, N * 4, M, N * 4, (int2)(n0 + j * 16, m0 + i * 8),
                (__private uint*)&acc[i][j]);
        }
    }
}

// ── Skinny GEMMs: M <= 16 rows (see gemm_kquant.cl's `gemm_kq_sk`) ─────────────────────────────
// DPAS B operand for 16 values of one sub-block half: `q` holds the 32 nibbles (low = values
// 0..15, high = 16..31), `h` picks the half; codebook pairs from `lut`, then the block scale.
inline int8 iq4_b(uint4 q, const int h, uint s2, __local const uint* lut) {
    uint4 x = h ? ((q >> 4) & 0x0F0F0F0Fu) : (q & 0x0F0F0F0Fu);
    uint8 v;
    v.s0 = lut[(x.s0 & 0xFu) | ((x.s0 >> 4) & 0xF0u)];
    v.s1 = lut[((x.s0 >> 16) & 0xFu) | ((x.s0 >> 20) & 0xF0u)];
    v.s2 = lut[(x.s1 & 0xFu) | ((x.s1 >> 4) & 0xF0u)];
    v.s3 = lut[((x.s1 >> 16) & 0xFu) | ((x.s1 >> 20) & 0xF0u)];
    v.s4 = lut[(x.s2 & 0xFu) | ((x.s2 >> 4) & 0xF0u)];
    v.s5 = lut[((x.s2 >> 16) & 0xFu) | ((x.s2 >> 20) & 0xF0u)];
    v.s6 = lut[(x.s3 & 0xFu) | ((x.s3 >> 4) & 0xF0u)];
    v.s7 = lut[((x.s3 >> 16) & 0xFu) | ((x.s3 >> 20) & 0xF0u)];
    return scale_b(v, s2);
}

// One 16-k step of the skinny tile: A rows 0..8*mt at column k against B operands b0, b1.
inline void sk_step(__global const half* A, int M, int K, int k, const int mt, int8 b0, int8 b1,
                    float8* acc) {
    short8 a[2];
    if (mt == 1) {
        intel_sub_group_2d_block_read_16b_8r16x1c((__global void*)A, K * 2, M, K * 2, (int2)(k, 0),
                                                  (__private ushort*)&a[0]);
    } else {
        ushort a16[16];
        intel_sub_group_2d_block_read_16b_16r16x1c((__global void*)A, K * 2, M, K * 2, (int2)(k, 0),
                                                   a16);
        a[0] = as_short8(vload8(0, a16));
        a[1] = as_short8(vload8(1, a16));
    }
    for (int i = 0; i < mt; i++) {
        acc[2 * i] = intel_sub_group_f16_f16_matrix_mad_k16(a[i], b0, acc[2 * i]);
        acc[2 * i + 1] = intel_sub_group_f16_f16_matrix_mad_k16(a[i], b1, acc[2 * i + 1]);
    }
}

inline void sk_store(__global float* C, int M, int N, int n0, const int mt, float8* acc) {
    __global float* Cs = C + (size_t)get_group_id(1) * M * N;
    for (int i = 0; i < mt; i++) {
        for (int j = 0; j < 2; j++) {
            intel_sub_group_2d_block_write_32b_8r16x1c((__global void*)Cs, N * 4, M, N * 4,
                                                       (int2)(n0 + j * 16, i * 8),
                                                       (__private uint*)&acc[2 * i + j]);
        }
    }
}

inline void gemm_iq4xs_sk(__global const half* A, __global const uchar* W, __global float* C,
                          int M, int N, int K, int nsplit, const int mt, __local uint* lut) {
    const int sg = get_sub_group_id();
    const int lane = get_sub_group_local_id();
    const int n0 = get_group_id(0) * 256 + sg * 32;
    if (n0 >= N)
        return;
    const int nsb = K / QK;
    const int per = (nsb + nsplit - 1) / nsplit;
    const int s0 = get_group_id(1) * per;
    const int s1 = min(nsb, s0 + per);
    const size_t rowb = (size_t)nsb * BLK;
    __global const uchar* row0 = W + (size_t)(n0 + lane) * rowb;
    __global const uchar* row1 = n0 + 16 < N ? row0 + 16 * rowb : row0;
    float8 acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int s = s0; s < s1; s++) {
        __global const uchar* blk[2] = {row0 + s * BLK, row1 + s * BLK};
        uint2 hd[2];
        hd[0] = vload2(0, (__global const uint*)blk[0]);
        hd[1] = vload2(0, (__global const uint*)blk[1]);
        __attribute__((opencl_unroll_hint))
        for (int ib = 0; ib < 8; ib++) {
            uint4 q[2];
            uint s2[2];
            for (int j = 0; j < 2; j++) {
                q[j] = vload4(0, (__global const uint*)(blk[j] + 8 + ib * 16));
                const int ls = ((hd[j].y >> (4 * ib)) & 0xF) | (((hd[j].x >> (16 + 2 * ib)) & 3) << 4);
                const half dl = as_half((ushort)hd[j].x) * (half)(ls - 32);
                s2[j] = as_uint((half2)(dl));
            }
            __attribute__((opencl_unroll_hint))
            for (int h = 0; h < 2; h++) {
                sk_step(A, M, K, s * QK + ib * 32 + h * 16, mt, iq4_b(q[0], h, s2[0], lut),
                        iq4_b(q[1], h, s2[1], lut), acc);
            }
        }
    }
    sk_store(C, M, N, n0, mt, acc);
}

// IQ4_NL (18-byte blocks of 32), walked in dword-aligned block pairs as `gemm_iq4nl`; the K split
// is over pairs.
inline void gemm_iq4nl_sk(__global const half* A, __global const uchar* W, __global float* C,
                          int M, int N, int K, int nsplit, const int mt, __local uint* lut) {
    const int sg = get_sub_group_id();
    const int lane = get_sub_group_local_id();
    const int n0 = get_group_id(0) * 256 + sg * 32;
    if (n0 >= N)
        return;
    const int npair = K / 64;
    const int per = (npair + nsplit - 1) / nsplit;
    const int p0 = get_group_id(1) * per;
    const int p1 = min(npair, p0 + per);
    const size_t rowb = (size_t)(K / 32) * 18;
    __global const uchar* rows[2];
    rows[0] = W + (size_t)(n0 + lane) * rowb;
    rows[1] = n0 + 16 < N ? rows[0] + 16 * rowb : rows[0];
    float8 acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int bp = p0; bp < p1; bp++) {
        __attribute__((opencl_unroll_hint))
        for (int odd = 0; odd < 2; odd++) {
            const int b = 2 * bp + odd;
            uint4 q[2];
            uint s2[2];
            for (int j = 0; j < 2; j++) {
                __global const uchar* blk = rows[j] + (size_t)b * 18;
                s2[j] = as_uint((half2)(ld_d(blk, odd)));
                q[j] = ld16(blk + 2, !odd);
            }
            __attribute__((opencl_unroll_hint))
            for (int h = 0; h < 2; h++) {
                sk_step(A, M, K, b * 32 + h * 16, mt, iq4_b(q[0], h, s2[0], lut),
                        iq4_b(q[1], h, s2[1], lut), acc);
            }
        }
    }
    sk_store(C, M, N, n0, mt, acc);
}

// Q8_0 (34-byte blocks of 32: f16 d, then 32 int8), walked in dword-aligned block pairs (K % 64
// == 0) as IQ4_NL is. No codebook: int8 -> half is exact, then the block scale.
inline void gemm_q8_0_sk(__global const half* A, __global const uchar* W, __global float* C,
                         int M, int N, int K, int nsplit, const int mt, __local uint* lut) {
    const int sg = get_sub_group_id();
    const int lane = get_sub_group_local_id();
    const int n0 = get_group_id(0) * 256 + sg * 32;
    if (n0 >= N)
        return;
    const int npair = K / 64;
    const int per = (npair + nsplit - 1) / nsplit;
    const int p0 = get_group_id(1) * per;
    const int p1 = min(npair, p0 + per);
    const size_t rowb = (size_t)(K / 32) * 34;
    __global const uchar* rows[2];
    rows[0] = W + (size_t)(n0 + lane) * rowb;
    rows[1] = n0 + 16 < N ? rows[0] + 16 * rowb : rows[0];
    float8 acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int bp = p0; bp < p1; bp++) {
        __attribute__((opencl_unroll_hint))
        for (int odd = 0; odd < 2; odd++) {
            const int b = 2 * bp + odd;
            half d[2];
            uint4 q[2][2];
            for (int j = 0; j < 2; j++) {
                __global const uchar* blk = rows[j] + (size_t)b * 34;
                d[j] = ld_d(blk, odd);
                q[j][0] = ld16(blk + 2, !odd);
                q[j][1] = ld16(blk + 18, !odd);
            }
            __attribute__((opencl_unroll_hint))
            for (int h = 0; h < 2; h++) {
                int8 bb[2];
                for (int j = 0; j < 2; j++)
                    bb[j] = as_int8(convert_half16(as_char16(q[j][h])) * d[j]);
                sk_step(A, M, K, b * 32 + h * 16, mt, bb[0], bb[1], acc);
            }
        }
    }
    sk_store(C, M, N, n0, mt, acc);
}

#define IQ4_SKINNY(name, body, mt)                                                                 \
    __attribute__((intel_reqd_sub_group_size(16)))                                                 \
    __attribute__((reqd_work_group_size(128, 1, 1)))                                               \
    __kernel void name(__global const half* A, __global const uchar* W, __global float* C, int M,  \
                       int N, int K, int nsplit) {                                                 \
        __local uint lut[256];                                                                     \
        for (int i = get_local_id(0); i < 256; i += 128)                                           \
            lut[i] = as_uint((half2)((half)KV[i & 15], (half)KV[i >> 4]));                         \
        barrier(CLK_LOCAL_MEM_FENCE);                                                              \
        body(A, W, C, M, N, K, nsplit, mt, lut);                                                   \
    }
IQ4_SKINNY(gemm_iq4xs_m8, gemm_iq4xs_sk, 1)
IQ4_SKINNY(gemm_iq4xs_m16, gemm_iq4xs_sk, 2)
IQ4_SKINNY(gemm_iq4nl_m8, gemm_iq4nl_sk, 1)
IQ4_SKINNY(gemm_iq4nl_m16, gemm_iq4nl_sk, 2)
IQ4_SKINNY(gemm_q8_0_m8, gemm_q8_0_sk, 1)
IQ4_SKINNY(gemm_q8_0_m16, gemm_q8_0_sk, 2)
