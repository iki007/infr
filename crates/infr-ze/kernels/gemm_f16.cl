// ocloc-options: -cl-intel-256-GRF-per-thread
// Probe: C[M][N] (f32) = A[M][K] (f16, row-major) . W[N][K]^T (f16, row-major: a weight matrix).
// DPAS (8x16x16 per instruction, 16-lane subgroups) fed straight from 2D block loads, no shared
// memory: each subgroup keeps a 32 x 64 C tile in registers (4 x 4 accumulators). A comes in as
// 32-row x 32-k blocks (row-major read: each lane gets one k column of 32 rows); W as transposed
// 32-bit reads of 16 rows x 8 words (each lane gets one row of W = one n, 16 k as 8 k-pairs), which
// is exactly DPAS's B operand. Workgroup: 8 subgroups as 4 (M) x 2 (N) = a 128 x 128 C tile.
// Requires M % 128 == 0, N % 128 == 0, K % 32 == 0.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define SG_M 32
#define SG_N 64
#define WG_SG_M 4
#define WG_SG_N 2

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void gemm_f16(__global const half* A, __global const half* W, __global float* C,
                       int M, int N, int K) {
    int sg = get_sub_group_id();
    // Group id 0 runs over M (fast), 1 over N: the M/128 workgroups that share a W column tile
    // run back to back, so W comes from DRAM once and from L2 for the rest.
    int m0 = get_group_id(0) * (SG_M * WG_SG_M) + (sg / WG_SG_N) * SG_M;
    int n0 = get_group_id(1) * (SG_N * WG_SG_N) + (sg % WG_SG_N) * SG_N;
    float8 acc[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;
    int a_bytes = K * 2;
    for (int k0 = 0; k0 < K; k0 += 32) {
        ushort a[64]; // [k block][32 rows]
        intel_sub_group_2d_block_read_16b_32r16x2c((__global void*)A, a_bytes, M, a_bytes,
                                                   (int2)(k0, m0), a);
        uint b[4][2][8]; // [n block][k block][k pair]
        for (int j = 0; j < 4; j++)
            for (int kb = 0; kb < 2; kb++)
                intel_sub_group_2d_block_read_transpose_32b_16r8x1c(
                    (__global void*)W, a_bytes, N, a_bytes, (int2)(k0 / 2 + kb * 8, n0 + j * 16),
                    b[j][kb]);
        for (int kb = 0; kb < 2; kb++)
            for (int i = 0; i < 4; i++) {
                short8 af = as_short8(vload8(0, (ushort*)&a[kb * 32 + i * 8]));
                for (int j = 0; j < 4; j++)
                    acc[i][j] = intel_sub_group_f16_f16_matrix_mad_k16(
                        af, as_int8(vload8(0, b[j][kb])), acc[i][j]);
            }
    }
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            intel_sub_group_2d_block_write_32b_8r16x1c((__global void*)C, N * 4, M, N * 4,
                                                       (int2)(n0 + j * 16, m0 + i * 8),
                                                       (private uint*)&acc[i][j]);
}
