// ocloc-options: -cl-intel-256-GRF-per-thread
// C[M,N] (f32) = A[M,K] (f16) . W[N,K]^T, W in GGUF Q4_K (144-byte superblocks of 256, row-major
// over K), DPAS on Xe2.
//
// Each 16-lane subgroup holds a 64x32 C tile (8x2 DPAS accumulators). Lane l of the subgroup owns
// W rows n0+l and n0+16+l: DPAS's B operand is 16 k-values of ONE column per lane, so each lane
// dequantizes its own rows' nibbles straight into that operand (`deq_b`: mask, widen, scale) —
// no shared memory, no barriers. A arrives as 32-row x 16-k 2D block reads. 8 subgroups per workgroup,
// 4 (M) x 2 (N): a 256x64 tile. N must be a multiple of 64 and K of 256; M is bounds-checked by
// the 2D block reads/writes.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define SG_M 64
#define SG_N 32
#define WG_M 256
#define WG_N 64
#define QK 256
#define BLK 144

// One DPAS B operand (16 k of this lane's W row, 8 dwords of k-pairs) from four dwords of masked
// nibbles (byte e of `nib` dword d = k 4d+e): row i = half2(q[2i], q[2i+1]) * s + m, with `s2` /
// `m2` the lane's scale and min twice (half2). Written in vISA because OpenCL C's per-lane half
// vectors live de-interleaved in registers while DPAS wants each lane's k-pairs packed in one
// dword: IGC shuffles them back and forth (~16 instructions per B row); this is 4 per row.
// `0x6400 | q` is the half 1024 + q exactly, so `- 1024` (exact) leaves q. The intermediates live
// in asm-local registers and only the final mads write `out`: IGC's inline asm has no early-clobber
// constraint, so `out` may share registers with the inputs.
inline int8 deq_b(uint4 nib, uint s2, uint m2) {
    uint8 out;
    __asm__("{\n"
            ".decl XB v_type=G type=ub num_elts=256 align=GRF alias=<%1, 0>\n"
            ".decl OH v_type=G type=hf num_elts=256 align=GRF alias=<%0, 0>\n"
            ".decl TW v_type=G type=uw num_elts=256 align=GRF\n"
            ".decl TH v_type=G type=hf num_elts=256 align=GRF alias=<TW, 0>\n"
            ".decl SD v_type=G type=ud num_elts=32 align=GRF\n"
            ".decl SH v_type=G type=hf num_elts=64 align=GRF alias=<SD, 0>\n"
            "mov (M1_NM, 16) SD(0,0)<1> %2(0,0)<1;1,0>\n"
            "mov (M1_NM, 16) SD(1,0)<1> %3(0,0)<1;1,0>\n"
            "mov (M1_NM, 32) TW(0,0)<1> XB(0,0)<4;2,1>\n"
            "mov (M1_NM, 32) TW(1,0)<1> XB(0,2)<4;2,1>\n"
            "mov (M1_NM, 32) TW(2,0)<1> XB(1,0)<4;2,1>\n"
            "mov (M1_NM, 32) TW(3,0)<1> XB(1,2)<4;2,1>\n"
            "mov (M1_NM, 32) TW(4,0)<1> XB(2,0)<4;2,1>\n"
            "mov (M1_NM, 32) TW(5,0)<1> XB(2,2)<4;2,1>\n"
            "mov (M1_NM, 32) TW(6,0)<1> XB(3,0)<4;2,1>\n"
            "mov (M1_NM, 32) TW(7,0)<1> XB(3,2)<4;2,1>\n"
            "or (M1_NM, 32) TW(0,0)<1> TW(0,0)<1;1,0> 0x6400:uw\n"
            "or (M1_NM, 32) TW(1,0)<1> TW(1,0)<1;1,0> 0x6400:uw\n"
            "or (M1_NM, 32) TW(2,0)<1> TW(2,0)<1;1,0> 0x6400:uw\n"
            "or (M1_NM, 32) TW(3,0)<1> TW(3,0)<1;1,0> 0x6400:uw\n"
            "or (M1_NM, 32) TW(4,0)<1> TW(4,0)<1;1,0> 0x6400:uw\n"
            "or (M1_NM, 32) TW(5,0)<1> TW(5,0)<1;1,0> 0x6400:uw\n"
            "or (M1_NM, 32) TW(6,0)<1> TW(6,0)<1;1,0> 0x6400:uw\n"
            "or (M1_NM, 32) TW(7,0)<1> TW(7,0)<1;1,0> 0x6400:uw\n"
            "add (M1_NM, 32) TH(0,0)<1> TH(0,0)<1;1,0> 0xE400:hf\n"
            "add (M1_NM, 32) TH(1,0)<1> TH(1,0)<1;1,0> 0xE400:hf\n"
            "add (M1_NM, 32) TH(2,0)<1> TH(2,0)<1;1,0> 0xE400:hf\n"
            "add (M1_NM, 32) TH(3,0)<1> TH(3,0)<1;1,0> 0xE400:hf\n"
            "add (M1_NM, 32) TH(4,0)<1> TH(4,0)<1;1,0> 0xE400:hf\n"
            "add (M1_NM, 32) TH(5,0)<1> TH(5,0)<1;1,0> 0xE400:hf\n"
            "add (M1_NM, 32) TH(6,0)<1> TH(6,0)<1;1,0> 0xE400:hf\n"
            "add (M1_NM, 32) TH(7,0)<1> TH(7,0)<1;1,0> 0xE400:hf\n"
            "mad (M1_NM, 32) OH(0,0)<1> TH(0,0)<1;1,0> SH(0,0)<1;1,0> SH(1,0)<1;1,0>\n"
            "mad (M1_NM, 32) OH(1,0)<1> TH(1,0)<1;1,0> SH(0,0)<1;1,0> SH(1,0)<1;1,0>\n"
            "mad (M1_NM, 32) OH(2,0)<1> TH(2,0)<1;1,0> SH(0,0)<1;1,0> SH(1,0)<1;1,0>\n"
            "mad (M1_NM, 32) OH(3,0)<1> TH(3,0)<1;1,0> SH(0,0)<1;1,0> SH(1,0)<1;1,0>\n"
            "mad (M1_NM, 32) OH(4,0)<1> TH(4,0)<1;1,0> SH(0,0)<1;1,0> SH(1,0)<1;1,0>\n"
            "mad (M1_NM, 32) OH(5,0)<1> TH(5,0)<1;1,0> SH(0,0)<1;1,0> SH(1,0)<1;1,0>\n"
            "mad (M1_NM, 32) OH(6,0)<1> TH(6,0)<1;1,0> SH(0,0)<1;1,0> SH(1,0)<1;1,0>\n"
            "mad (M1_NM, 32) OH(7,0)<1> TH(7,0)<1;1,0> SH(0,0)<1;1,0> SH(1,0)<1;1,0>\n"
            "}\n"
            : "=rw"(out)
            : "rw"(nib), "rw"(s2), "rw"(m2));
    return as_int8(out);
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void gemm_q4k(__global const half* A, __global const uchar* W, __global float* C,
                       int M, int N, int K) {
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
        half8 S[2], Mn[2];
        __global const uchar* blk[2] = {row0 + s * BLK, row1 + s * BLK};
        for (int j = 0; j < 2; j++) {
            // d, dmin, then scales[12]: 6-bit (scale, min) for 8 sub-blocks of 32.
            uint4 hd = vload4(0, (__global const uint*)blk[j]);
            half2 dd = as_half2(hd.s0);
            uint scl = hd.s1 & 0x3F3F3F3Fu;
            uint mnl = hd.s2 & 0x3F3F3F3Fu;
            uint sch = (hd.s3 & 0x0F0F0F0Fu) | ((hd.s1 >> 2) & 0x30303030u);
            uint mnh = ((hd.s3 >> 4) & 0x0F0F0F0Fu) | ((hd.s2 >> 2) & 0x30303030u);
            S[j] = convert_half8(as_uchar8((uint2)(scl, sch))) * dd.s0;
            Mn[j] = convert_half8(as_uchar8((uint2)(mnl, mnh))) * dd.s1;
        }
        __attribute__((opencl_unroll_hint))
        for (int c = 0; c < 4; c++) {
            // This chunk's two sub-blocks: (scale, min) of each, twice per dword (half2).
            uint s2[2][2], m2[2][2];
            for (int j = 0; j < 2; j++) {
                for (int h = 0; h < 2; h++) {
                    s2[j][h] = as_uint((half2)(S[j][2 * c + h]));
                    m2[j][h] = as_uint((half2)(-Mn[j][2 * c + h]));
                }
            }
            uint8 q[2];
            q[0] = vload8(0, (__global const uint*)(blk[0] + 16 + c * 32));
            q[1] = vload8(0, (__global const uint*)(blk[1] + 16 + c * 32));
            __attribute__((opencl_unroll_hint))
            for (int kb = 0; kb < 4; kb++) {
                const int k = s * QK + c * 64 + kb * 16;
                ushort a0[32], a1[32];
                intel_sub_group_2d_block_read_16b_32r16x1c(
                    (__global void*)A, K * 2, M, K * 2, (int2)(k, m0), a0);
                intel_sub_group_2d_block_read_16b_32r16x1c(
                    (__global void*)A, K * 2, M, K * 2, (int2)(k, m0 + 32), a1);
                int8 b[2];
                for (int j = 0; j < 2; j++) {
                    uint4 w = (kb & 1) ? q[j].s4567 : q[j].s0123;
                    uint4 nib = (kb & 2) ? ((w >> 4) & 0x0F0F0F0Fu) : (w & 0x0F0F0F0Fu);
                    b[j] = deq_b(nib, s2[j][kb >> 1], m2[j][kb >> 1]);
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
