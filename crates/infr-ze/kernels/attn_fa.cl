// ocloc-options: -cl-intel-256-GRF-per-thread
// Flash attention prefill on Xe2 DPAS: O[q, h, :] = softmax(scale * Q[q, h, :] . K[:, kvh, :]^T
// + mask) . V[:, kvh, :], with Q f16 [q_len, nh, HD], K/V f16 caches [kv_len, nkv, HD] (row =
// position), O f32 [q_len, nh, HD] — the semantics of the Vulkan `attn_xmx_fa.comp` (causal from
// `pos`, optional sliding window, GQA kvh = h / (nh / nkv), scale 0 -> 1/sqrt(HD), a fully masked
// row -> 0), without its ring-cache mapping.
//
// A 16-lane subgroup owns 8 query rows of one head and half of their 8 x HD output in registers. Per
// block of 32 keys: S = Q . K^T (K arrives as transposed 2D block reads: lane = key, which is DPAS's
// B operand), mask + online softmax (one cross-lane max per row; the row sums stay per-lane until
// the end), then P . V — S's accumulator layout (lane = key, 8 rows) is exactly DPAS's A operand,
// and V arrives through VNNI-transform 2D block reads. A workgroup is 8 subgroups = 32 query rows x
// both output halves of one head, so its K/V blocks are shared through L1; adjacent workgroups are
// the other heads of the same KV group.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define HD 256
#define DB (HD / 16)
// Output d-blocks per subgroup: the head dim is split across two subgroups (each recomputes the
// full S = Q.K^T — QK is half the FLOPs — but keeps only half of O in registers: the whole 8 x 256
// O plus S spilled ~140 registers).
#define OB (DB / 2)
#define BR 8
// Match Vulkan's softmax block width: changing the rescaling points also changes how P rounds
// to f16, enough to perturb later greedy tokens on the 4B with 64-key blocks.
#define BK 32
#define NB (BK / 16)
#define DEAD (-1e30f)

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void attn_fa256(__global const half* Q, __global const half* Kc, __global const half* Vc,
                         __global float* O, int q_len, int kv_len, int nh, int nkv, int pos,
                         int window, float scale) {
    const int sg = get_sub_group_id();
    const int lane = get_sub_group_local_id();
    // Workgroup = 32 query rows x one head: subgroup sg owns rows r0..r0+7 and output half dh.
    const int dh = sg & 1;
    const int h = get_group_id(0) % nh;
    const int r0 = (get_group_id(0) / nh) * (4 * BR) + (sg >> 1) * BR;
    if (r0 >= q_len)
        return;
    const int kvh = h / (nh / nkv);
    if (scale <= 0.0f)
        scale = rsqrt((float)HD);
    __global const half* Qh = Q + h * HD;
    __global const half* Kh = Kc + kvh * HD;
    __global const half* Vh = Vc + kvh * HD;
    const int qpitch = nh * HD * 2, kpitch = nkv * HD * 2;

    float8 o[OB];
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < OB; i++)
        o[i] = 0.0f;
    float8 m = DEAD, l = 0.0f;

    // Keys this tile can see: the last row (pos + r0 + 7) bounds the end; the first row's window
    // bounds the start (the mask handles the rows between).
    const int kend = min(kv_len, pos + r0 + BR);
    int kbeg = 0;
    if (window > 0 && pos + r0 + 1 > window)
        kbeg = (pos + r0 + 1 - window) & ~(BK - 1);
    for (int k0 = kbeg; k0 < kend; k0 += BK) {
        float8 s[NB];
        __attribute__((opencl_unroll_hint)) for (int n = 0; n < NB; n++)
            s[n] = 0.0f;
        __attribute__((opencl_unroll_hint)) for (int db = 0; db < DB; db++) {
            short8 qa;
            intel_sub_group_2d_block_read_16b_8r16x1c((__global void*)Qh, HD * 2, q_len, qpitch,
                                                      (int2)(db * 16, r0), (__private ushort*)&qa);
            __attribute__((opencl_unroll_hint)) for (int n = 0; n < NB; n++) {
                int8 kb;
                intel_sub_group_2d_block_read_transpose_32b_16r8x1c(
                    (__global void*)Kh, HD * 2, kv_len, kpitch, (int2)(db * 8, k0 + 16 * n),
                    (__private uint*)&kb);
                s[n] = intel_sub_group_f16_f16_matrix_mad_k16(qa, kb, s[n]);
            }
        }
        // Scale, mask (causal / window / past kv_len), and the block's row max.
        const bool edge = (k0 + BK > pos + r0) || (k0 + BK > kv_len) || (window > 0);
        float8 mx = DEAD;
        __attribute__((opencl_unroll_hint)) for (int n = 0; n < NB; n++) {
            float8 v = s[n] * scale;
            if (edge) {
                const int key = k0 + 16 * n + lane;
                __attribute__((opencl_unroll_hint)) for (int e = 0; e < BR; e++) {
                    const int qp = pos + r0 + e;
                    const bool dead = key >= kv_len || key > qp ||
                                      (window > 0 && qp + 1 > window && key < qp + 1 - window);
                    if (dead)
                        v[e] = DEAD;
                }
            }
            s[n] = v;
            mx = fmax(mx, v);
        }
        float8 mnew;
        __attribute__((opencl_unroll_hint)) for (int e = 0; e < BR; e++)
            mnew[e] = fmax(m[e], sub_group_reduce_max(mx[e]));
        const float8 alpha = exp(m - mnew);
        m = mnew;
        l *= alpha;
        __attribute__((opencl_unroll_hint)) for (int i = 0; i < OB; i++)
            o[i] *= alpha;
        __attribute__((opencl_unroll_hint)) for (int n = 0; n < NB; n++) {
            float8 p;
            __attribute__((opencl_unroll_hint)) for (int e = 0; e < BR; e++)
                p[e] = s[n][e] > -1e29f ? exp(s[n][e] - m[e]) : 0.0f;
            l += p;
            const short8 pa = as_short8(convert_half8(p));
            __attribute__((opencl_unroll_hint)) for (int db = 0; db < OB; db++) {
                int8 vb;
                intel_sub_group_2d_block_read_transform_16b_16r16x1c(
                    (__global void*)Vh, HD * 2, kv_len, kpitch,
                    (int2)((dh * OB + db) * 16, k0 + 16 * n), (__private uint*)&vb);
                o[db] = intel_sub_group_f16_f16_matrix_mad_k16(pa, vb, o[db]);
            }
        }
    }
    float8 inv;
    __attribute__((opencl_unroll_hint)) for (int e = 0; e < BR; e++) {
        const float t = sub_group_reduce_add(l[e]);
        inv[e] = t > 0.0f ? 1.0f / t : 0.0f;
    }
    __global float* Oh = O + h * HD;
    __attribute__((opencl_unroll_hint)) for (int db = 0; db < OB; db++) {
        float8 v = o[db] * inv;
        intel_sub_group_2d_block_write_32b_8r16x1c((__global void*)Oh, HD * 4, q_len, nh * HD * 4,
                                                   (int2)((dh * OB + db) * 16, r0),
                                                   (__private uint*)&v);
    }
}
