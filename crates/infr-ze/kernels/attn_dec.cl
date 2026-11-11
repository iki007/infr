// ocloc-options: -cl-intel-256-GRF-per-thread
// Split-K flash attention for a verify-sized batch (rows <= 16) on Xe2 DPAS: the decode-shaped twin
// of `attn_fa.cl`'s prefill kernel. Q f16 [rows, nh, 256], K/V f16 caches [kv_len.., nkv, 256]
// (row = position, no ring), causal from `pos` (row r sees keys <= pos + r), GQA kvh = h /
// (nh / nkv), `scale` 0 -> 1/16.
//
// The key range is cut into `chunk`-key pieces so a 32k-token context still fills the card: one
// 32-item workgroup per (head, chunk, 8-row tile) — its two subgroups hold the two halves of the
// 8 x 256 output — runs `attn_fa256`'s block loop over its chunk and writes the UNNORMALIZED partial
// (o, row max m, row sum l) to slice `chunk` of `Po` / `Pml`; `attn_dec_combine` merges the slices.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define HD 256
#define DB (HD / 16)
#define OB (DB / 2)
#define BR 8
#define BK 32
#define NB (BK / 16)
#define DEAD (-1e30f)

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(32, 1, 1)))
__kernel void attn_dec256(__global const half* Q, __global const half* Kc, __global const half* Vc,
                          __global float* Po, __global float* Pml, int q_len, int kv_len, int nh,
                          int nkv, int pos, int chunk, float scale) {
    const int dh = get_sub_group_id();
    const int lane = get_sub_group_local_id();
    const int h = get_group_id(0);
    const int c = get_group_id(1);
    const int r0 = get_group_id(2) * BR;
    const int kvh = h / (nh / nkv);
    if (scale <= 0.0f)
        scale = rsqrt((float)HD);
    __global const half* Qh = Q + h * HD;
    __global const half* Kh = Kc + kvh * HD;
    __global const half* Vh = Vc + kvh * HD;
    const int qpitch = nh * HD * 2, kpitch = nkv * HD * 2;

    float8 o[OB];
    for (int i = 0; i < OB; i++)
        o[i] = 0.0f;
    float8 m = DEAD, l = 0.0f;
    const int kbeg = c * chunk;
    const int kend = min(min(kv_len, kbeg + chunk), pos + r0 + BR);
    for (int k0 = kbeg; k0 < kend; k0 += BK) {
        float8 s[NB];
        for (int n = 0; n < NB; n++)
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
        float8 mx = DEAD;
        __attribute__((opencl_unroll_hint)) for (int n = 0; n < NB; n++) {
            float8 v = s[n] * scale;
            const int key = k0 + 16 * n + lane;
            __attribute__((opencl_unroll_hint)) for (int e = 0; e < BR; e++) {
                if (key >= kend || key > pos + r0 + e)
                    v[e] = DEAD;
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
        for (int i = 0; i < OB; i++)
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
    // Partials: lane = output column within a 16-block, the 8 values = the tile's 8 rows.
    float8 lt;
    for (int e = 0; e < BR; e++)
        lt[e] = sub_group_reduce_add(l[e]);
    for (int e = 0; e < BR; e++) {
        const int r = r0 + e;
        if (r >= q_len)
            break;
        const size_t unit = ((size_t)c * q_len + r) * nh + h;
        for (int db = 0; db < OB; db++)
            Po[unit * HD + (dh * OB + db) * 16 + lane] = o[db][e];
        if (dh == 0 && lane == 0) {
            Pml[unit * 2] = m[e];
            Pml[unit * 2 + 1] = lt[e];
        }
    }
}

// O[r, h, :] = sum_c exp(m_c - M) o_c / sum_c exp(m_c - M) l_c over the chunks; a row no chunk saw
// (all masked) -> 0. One 256-item workgroup per (row, head).
__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void attn_dec_combine(__global const float* Po, __global const float* Pml,
                               __global float* O, int q_len, int nh, int nchunk) {
    const int unit = get_group_id(0); // r * nh + h
    const int d = get_local_id(0);
    float M = DEAD;
    for (int c = 0; c < nchunk; c++)
        M = fmax(M, Pml[((size_t)c * q_len * nh + unit) * 2]);
    float L = 0.0f, acc = 0.0f;
    for (int c = 0; c < nchunk; c++) {
        const size_t u = (size_t)c * q_len * nh + unit;
        const float mc = Pml[u * 2];
        if (mc <= -1e29f)
            continue;
        const float w = exp(mc - M);
        L += w * Pml[u * 2 + 1];
        acc += w * Po[u * HD + d];
    }
    O[(size_t)unit * HD + d] = L > 0.0f ? acc / L : 0.0f;
}

// Head-dim-generic merge of the chunk partials (`attn_dec256` / `attn_ring128`): `hd` columns,
// 256 items striding over them. One workgroup per (row, head).
__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void attn_combine_hd(__global const float* Po, __global const float* Pml,
                              __global float* O, int q_len, int nh, int nchunk, int hd) {
    const int unit = get_group_id(0);
    float M = DEAD;
    for (int c = 0; c < nchunk; c++)
        M = fmax(M, Pml[((size_t)c * q_len * nh + unit) * 2]);
    for (int d = get_local_id(0); d < hd; d += 256) {
        float L = 0.0f, acc = 0.0f;
        for (int c = 0; c < nchunk; c++) {
            const size_t u = (size_t)c * q_len * nh + unit;
            const float mc = Pml[u * 2];
            if (mc <= -1e29f)
                continue;
            const float w = exp(mc - M);
            L += w * Pml[u * 2 + 1];
            acc += w * Po[u * hd + d];
        }
        O[(size_t)unit * hd + d] = L > 0.0f ? acc / L : 0.0f;
    }
}

// The DFlash drafter's block attention: head dim 128, every query row sees the same key range
// [lo, kv_len) (`AttnMask::Canvas`), and the K/V caches are a RING of `cap` rows (key j at row
// j % cap) with cap % 32 == 0, so a 32-key block starting at a multiple of 32 never straddles the
// wrap. Chunks start at lo rounded down to 32. Same split-K partials as `attn_dec256`.
#define HD2 128
#define DB2 (HD2 / 16)
#define OB2 (DB2 / 2)
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(32, 1, 1)))
__kernel void attn_ring128(__global const half* Q, __global const half* Kc, __global const half* Vc,
                           __global float* Po, __global float* Pml, int q_len, int kv_len, int nh,
                           int nkv, int lo, int cap, int chunk, float scale) {
    const int dh = get_sub_group_id();
    const int lane = get_sub_group_local_id();
    const int h = get_group_id(0);
    const int c = get_group_id(1);
    const int r0 = get_group_id(2) * BR;
    const int kvh = h / (nh / nkv);
    if (scale <= 0.0f)
        scale = rsqrt((float)HD2);
    __global const half* Qh = Q + h * HD2;
    __global const half* Kh = Kc + kvh * HD2;
    __global const half* Vh = Vc + kvh * HD2;
    const int qpitch = nh * HD2 * 2, kpitch = nkv * HD2 * 2;

    float8 o[OB2];
    for (int i = 0; i < OB2; i++)
        o[i] = 0.0f;
    float8 m = DEAD, l = 0.0f;
    const int kbase = lo & ~(BK - 1);
    const int kbeg = kbase + c * chunk;
    const int kend = min(kv_len, kbeg + chunk);
    for (int k0 = kbeg; k0 < kend; k0 += BK) {
        const int row0 = k0 % cap;
        float8 s[NB];
        for (int n = 0; n < NB; n++)
            s[n] = 0.0f;
        __attribute__((opencl_unroll_hint)) for (int db = 0; db < DB2; db++) {
            short8 qa;
            intel_sub_group_2d_block_read_16b_8r16x1c((__global void*)Qh, HD2 * 2, q_len, qpitch,
                                                      (int2)(db * 16, r0), (__private ushort*)&qa);
            __attribute__((opencl_unroll_hint)) for (int n = 0; n < NB; n++) {
                int8 kb;
                intel_sub_group_2d_block_read_transpose_32b_16r8x1c(
                    (__global void*)Kh, HD2 * 2, cap, kpitch, (int2)(db * 8, row0 + 16 * n),
                    (__private uint*)&kb);
                s[n] = intel_sub_group_f16_f16_matrix_mad_k16(qa, kb, s[n]);
            }
        }
        float8 mx = DEAD;
        __attribute__((opencl_unroll_hint)) for (int n = 0; n < NB; n++) {
            float8 v = s[n] * scale;
            const int key = k0 + 16 * n + lane;
            if (key < lo || key >= kend)
                v = DEAD;
            s[n] = v;
            mx = fmax(mx, v);
        }
        float8 mnew;
        __attribute__((opencl_unroll_hint)) for (int e = 0; e < BR; e++)
            mnew[e] = fmax(m[e], sub_group_reduce_max(mx[e]));
        const float8 alpha = exp(m - mnew);
        m = mnew;
        l *= alpha;
        for (int i = 0; i < OB2; i++)
            o[i] *= alpha;
        __attribute__((opencl_unroll_hint)) for (int n = 0; n < NB; n++) {
            float8 p;
            __attribute__((opencl_unroll_hint)) for (int e = 0; e < BR; e++)
                p[e] = s[n][e] > -1e29f ? exp(s[n][e] - m[e]) : 0.0f;
            l += p;
            const short8 pa = as_short8(convert_half8(p));
            __attribute__((opencl_unroll_hint)) for (int db = 0; db < OB2; db++) {
                int8 vb;
                intel_sub_group_2d_block_read_transform_16b_16r16x1c(
                    (__global void*)Vh, HD2 * 2, cap, kpitch,
                    (int2)((dh * OB2 + db) * 16, row0 + 16 * n), (__private uint*)&vb);
                o[db] = intel_sub_group_f16_f16_matrix_mad_k16(pa, vb, o[db]);
            }
        }
    }
    float8 lt;
    for (int e = 0; e < BR; e++)
        lt[e] = sub_group_reduce_add(l[e]);
    for (int e = 0; e < BR; e++) {
        const int r = r0 + e;
        if (r >= q_len)
            break;
        const size_t unit = ((size_t)c * q_len + r) * nh + h;
        for (int db = 0; db < OB2; db++)
            Po[unit * HD2 + (dh * OB2 + db) * 16 + lane] = o[db][e];
        if (dh == 0 && lane == 0) {
            Pml[unit * 2] = m[e];
            Pml[unit * 2 + 1] = lt[e];
        }
    }
}
