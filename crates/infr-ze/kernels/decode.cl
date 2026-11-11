// The small-batch (verify-sized, 2..=16 rows) ops of a qwen35 layer, so a speculative verify runs
// as one Level Zero chain between its embedding gather and its argmax (`crates/infr-vulkan/src/ze.rs`,
// `docs/level-zero.md`). Each kernel mirrors the Vulkan shader it stands in for:
// `copy_strided.comp`, `store_f16.comp`, `silu_mul.comp` / `mul_sigmoid.comp`, `conv1d_silu.comp`,
// `deltanet.comp` (+ its per-row state history), `rmsnorm.comp -DGATE`, `qk_norm_rope.comp` /
// `qk_norm_rope_interleaved.comp`.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void copy_f32(__global const float* src, __global float* dst, int n) {
    const int i = get_global_id(0);
    if (i < n)
        dst[i] = src[i];
}

// For `rows` rows: dst[r*dst_stride + c] = src[r*src_stride + c], c < n.
__kernel void copy_strided(__global const float* src, __global float* dst, int rows, int n,
                           int src_stride, int dst_stride) {
    const int i = get_global_id(0);
    if (i >= rows * n)
        return;
    const int r = i / n, c = i - r * n;
    dst[(size_t)r * dst_stride + c] = src[(size_t)r * src_stride + c];
}

// A KV-cache row write: f32 -> f16 (`store_f16`), or an f16 copy.
__kernel void store_f16(__global const float* src, __global half* dst, int n) {
    const int i = get_global_id(0);
    if (i < n)
        dst[i] = (half)src[i];
}

__kernel void copy_f16(__global const half* src, __global half* dst, int n) {
    const int i = get_global_id(0);
    if (i < n)
        dst[i] = src[i];
}

// act 0: y = silu(gate) * up (contiguous gate). act 1: y = up * sigmoid(gate), the gate indexed as
// `mul_sigmoid.comp` does (strided / per-head interleaved blocks).
__kernel void gated_act(__global const float* gate, __global const float* up, __global float* y,
                        int n, int act, int gate_stride, int gate_row_width,
                        int gate_block_width) {
    const int i = get_global_id(0);
    if (i >= n)
        return;
    if (act == 0) {
        const float g = gate[i];
        y[i] = g / (1.0f + exp(-g)) * up[i];
        return;
    }
    float g;
    if (gate_block_width > 0) {
        const int total_cols = gate_stride / 2;
        const int headw = gate_block_width / 2;
        const int r = i / total_cols, c = i - r * total_cols;
        const int head = c / headw, off = c - head * headw;
        g = gate[(size_t)r * gate_stride + head * gate_block_width + headw + off];
    } else if (gate_stride > 0) {
        const int r = i / gate_row_width, c = i - r * gate_row_width;
        g = gate[(size_t)r * gate_stride + c];
    } else {
        g = gate[i];
    }
    y[i] = up[i] / (1.0f + exp(-g));
}

// Causal depthwise conv + SiLU over `rows` tokens in order, carrying the per-channel history
// (`state` [kconv-1, cc], `w` [cc, kconv]). One work-item per channel.
__kernel void conv1d_silu(__global const float* x, __global const float* w, __global float* state,
                          __global float* out, int rows, int cc, int kconv) {
    const int ch = get_global_id(0);
    if (ch >= cc)
        return;
    const int km1 = kconv - 1;
    for (int t = 0; t < rows; t++) {
        float acc = 0.0f;
        for (int k = 0; k < km1; k++)
            acc += state[k * cc + ch] * w[ch * kconv + k];
        const float cur = x[(size_t)t * cc + ch];
        acc += cur * w[ch * kconv + km1];
        out[(size_t)t * cc + ch] = acc / (1.0f + exp(-acc));
        for (int k = 0; k + 1 < km1; k++)
            state[k * cc + ch] = state[(k + 1) * cc + ch];
        if (km1 >= 1)
            state[(km1 - 1) * cc + ch] = cur;
    }
}

// Gated-DeltaNet recurrence over `rows` tokens (state carried), `deltanet.comp`'s layout: one
// 128-item workgroup per (value head, 16-column block); item (grp, col) keeps column d's key rows
// [grp*rpg, grp*rpg + rpg) in registers. `write_hist`: also store the state after every token to
// `hist` ([rows, nv*kd*vd]) — the speculative verify's rollback record.
#define COLS 16
#define GROUPS 8
#define RPG_MAX 16
__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void deltanet_hist(__global const float* q, __global const float* k,
                            __global const float* v, __global const float* blog,
                            __global const float* alpha, __global const float* acoef,
                            __global const float* dtbias, __global float* state,
                            __global float* outp, __global float* hist, int rows, int nv, int nk,
                            int kd, int vd, float eps, float qscale, int src_stride,
                            int write_hist) {
    __local float qn[128];
    __local float kn[128];
    __local float pkv[GROUPS * COLS];
    __local float po[GROUPS * COLS];
    const int t = get_local_id(0);
    const int col = t % COLS;
    const int grp = t / COLS;
    const int nblk = (vd + COLS - 1) / COLS;
    const int h = get_group_id(0) / nblk;
    const int d = (get_group_id(0) % nblk) * COLS + col;
    const int live = d < vd;
    const int rpg = (kd + GROUPS - 1) / GROUPS;
    const int k0 = grp * rpg;
    const int khid = h % nk;
    const int hqbase = khid * kd;
    const int hvbase = h * vd;
    const size_t sbase = (size_t)h * kd * vd;
    const size_t sn = (size_t)nv * kd * vd;

    float sreg[RPG_MAX];
    for (int r = 0; r < RPG_MAX; r++) {
        const int kk = k0 + r;
        sreg[r] = (live && r < rpg && kk < kd) ? state[sbase + (size_t)kk * vd + d] : 0.0f;
    }
    for (int tk = 0; tk < rows; tk++) {
        const int qbase = (src_stride > 0 ? tk * src_stride : tk * nk * kd) + hqbase;
        const int kbase = src_stride > 0 ? qbase + nk * kd : qbase;
        const int vin = (src_stride > 0 ? tk * src_stride + 2 * nk * kd : tk * nv * vd) + hvbase;
        const int vbase = tk * nv * vd + hvbase;
        const int ga = tk * nv + h;
        float qv = 0.0f, kv = 0.0f;
        if (t < kd) {
            qv = q[qbase + t];
            kv = k[kbase + t];
        }
        const float qsum = work_group_reduce_add(qv * qv);
        const float ksum = work_group_reduce_add(kv * kv);
        if (t < kd) {
            qn[t] = qv / sqrt(qsum + eps) * qscale;
            kn[t] = kv / sqrt(ksum + eps);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        const float beta = 1.0f / (1.0f + exp(-blog[ga]));
        const float zsp = alpha[ga] + dtbias[h];
        const float sp = fmax(zsp, 0.0f) + log(1.0f + exp(-fabs(zsp)));
        const float decay = exp(acoef[h] * sp);
        float pk = 0.0f;
        for (int r = 0; r < RPG_MAX; r++) {
            if (r < rpg && k0 + r < kd) {
                sreg[r] *= decay;
                pk += kn[k0 + r] * sreg[r];
            }
        }
        pkv[grp * COLS + col] = pk;
        barrier(CLK_LOCAL_MEM_FENCE);
        float kvv = 0.0f;
        for (int g = 0; g < GROUPS; g++)
            kvv += pkv[g * COLS + col];
        const float delta = live ? (v[vin + d] - kvv) * beta : 0.0f;
        float pq = 0.0f;
        for (int r = 0; r < RPG_MAX; r++) {
            if (r < rpg && k0 + r < kd) {
                sreg[r] += kn[k0 + r] * delta;
                pq += qn[k0 + r] * sreg[r];
            }
        }
        po[grp * COLS + col] = pq;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (grp == 0 && live) {
            float o = 0.0f;
            for (int g = 0; g < GROUPS; g++)
                o += po[g * COLS + col];
            outp[vbase + d] = o;
        }
        if (write_hist) {
            for (int r = 0; r < RPG_MAX; r++) {
                const int kk = k0 + r;
                if (live && r < rpg && kk < kd)
                    hist[(size_t)tk * sn + sbase + (size_t)kk * vd + d] = sreg[r];
            }
        }
        // The next token's first local write (qn/kn) follows this token's last reads of them
        // only after this barrier.
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int r = 0; r < RPG_MAX; r++) {
        const int kk = k0 + r;
        if (live && r < rpg && kk < kd)
            state[sbase + (size_t)kk * vd + d] = sreg[r];
    }
}

// y = x * inversesqrt(mean(x^2) + eps) * w * silu(z) over one `dim`-wide unit (a head) per
// workgroup.
__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void gated_rmsnorm(__global const float* x, __global const float* w,
                            __global const float* z, __global float* y, int dim, float eps) {
    const size_t base = (size_t)get_group_id(0) * dim;
    const int t = get_local_id(0);
    float ss = 0.0f;
    for (int i = t; i < dim; i += 128) {
        const float v = x[base + i];
        ss += v * v;
    }
    ss = work_group_reduce_add(ss);
    const float scale = rsqrt(ss / (float)dim + eps);
    for (int i = t; i < dim; i += 128) {
        const float g = z[base + i];
        y[base + i] = x[base + i] * scale * w[i] * (g / (1.0f + exp(-g)));
    }
}

// Per-head RMSNorm (x weight) then NeoX RoPE on the first `rope_dim` dims at position
// rope_pos + row, written as f16 to row `out_row0 + row` of `y` ([.., nheads, hd]). `x_stride` > 0:
// the interleaved q+gate input (head h of row r at r*x_stride + h*2*hd). One 64-item workgroup per
// (row, head).
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void qk_norm_rope(__global const float* x, __global const float* nw, __global half* y,
                           int nheads, int hd, int rope_dim, float theta, int rope_pos,
                           int out_row0, float eps, int x_stride) {
    __local float xs[512];
    const int t = get_local_id(0);
    const int h = get_group_id(0) % nheads;
    const int r = get_group_id(0) / nheads;
    const size_t in_base =
        x_stride > 0 ? (size_t)r * x_stride + (size_t)h * 2 * hd : ((size_t)r * nheads + h) * hd;
    float pss = 0.0f;
    for (int i = t; i < hd; i += 64) {
        const float v = x[in_base + i];
        xs[i] = v;
        pss += v * v;
    }
    const float ss = work_group_reduce_add(pss);
    const float scale = rsqrt(ss / (float)hd + eps);
    const int absp = rope_pos + r;
    const size_t ob = ((size_t)(out_row0 + r) * nheads + h) * hd;
    const int hf = rope_dim / 2;
    for (int p = t; p < hf; p += 64) {
        const float a = xs[p] * scale * nw[p];
        const float b = xs[p + hf] * scale * nw[p + hf];
        const float freq = pow(theta, -2.0f * (float)p / (float)rope_dim);
        const float ang = (float)absp * freq;
        const float s = sin(ang), co = cos(ang);
        y[ob + p] = (half)(a * co - b * s);
        y[ob + p + hf] = (half)(a * s + b * co);
    }
    for (int i = rope_dim + t; i < hd; i += 64)
        y[ob + i] = (half)(xs[i] * scale * nw[i]);
}

// DFlash2's dynamic two-tap block convolution (`Op::BlockConv`, `block_conv.comp`): per output
// element, sum over taps of (coeff[t, c/group + G*(tap + kernel*side)] + base[c + dim*(tap +
// kernel*side)]) * x[t - tap, c], the tap reading zero before its block's first row.
__kernel void block_conv(__global const float* x, __global const float* coeff,
                         __global const float* base, __global float* dst, int rows, int dim,
                         int group, int kernel_, int block, int side) {
    const int i = get_global_id(0);
    if (i >= rows * dim)
        return;
    const int t = i / dim, c = i - t * dim;
    const int ng = dim / group;
    const int cw = 2 * kernel_ * ng;
    const int taps = min(kernel_, block);
    const int tb = t % block;
    float acc = 0.0f;
    for (int tap = 0; tap < taps; tap++) {
        if (tb < tap)
            break;
        const int k = tap + kernel_ * side;
        const float w = coeff[t * cw + c / group + ng * k] + base[c + dim * k];
        acc += w * x[(t - tap) * dim + c];
    }
    dst[i] = acc;
}
