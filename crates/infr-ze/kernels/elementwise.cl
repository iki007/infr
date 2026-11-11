// The small ops between the prefill GEMMs, so a layer's GEMMs and the ops joining them run as one
// Level Zero chain (`crates/infr-vulkan/src/ze.rs`): RMSNorm, SwiGLU, the residual add, and the
// f32 -> f16 copy the GEMMs read. Same arithmetic as the Vulkan shaders they stand in for
// (`rmsnorm.comp`, `silu_mul_fused.comp`, `add.comp`, `store_f16`). An optional `y16` output is the
// next GEMM's f16 A copy (the Vulkan `-DY16` builds).
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// y[r, :] = x[r, :] * inversesqrt(mean(x[r, :]^2) + eps) * w; one 256-thread workgroup per row.
__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void rmsnorm(__global const float* x, __global const float* w, __global float* y,
                      __global half* y16, int dim, float eps, int write16) {
    const size_t base = (size_t)get_group_id(0) * dim;
    const int t = get_local_id(0);
    float ss = 0.0f;
    for (int i = t; i < dim; i += 256) {
        const float v = x[base + i];
        ss += v * v;
    }
    ss = work_group_reduce_add(ss);
    const float scale = rsqrt(ss / (float)dim + eps);
    for (int i = t; i < dim; i += 256) {
        const float v = x[base + i] * scale * w[i];
        y[base + i] = v;
        if (write16)
            y16[base + i] = (half)v;
    }
}

// Row r of `gu` is [gate (nff) | up (nff)]: y = silu(min(g, limit)) * clamp(u, -limit, limit)
// (the clamp only when `do_clamp`).
__kernel void swiglu(__global const float* gu, __global float* y, __global half* y16, int rows,
                     int nff, int do_clamp, float limit, int write16) {
    const size_t idx = get_global_id(0);
    if (idx >= (size_t)rows * nff)
        return;
    const size_t r = idx / nff, i = idx % nff;
    const size_t base = r * 2 * nff;
    float g = gu[base + i];
    float u = gu[base + nff + i];
    if (do_clamp) {
        g = fmin(g, limit);
        u = clamp(u, -limit, limit);
    }
    const float v = g / (1.0f + exp(-g)) * u;
    y[idx] = v;
    if (write16)
        y16[idx] = (half)v;
}

__kernel void add(__global const float* a, __global const float* b, __global float* dst, int n) {
    const size_t i = get_global_id(0);
    if (i < (size_t)n)
        dst[i] = a[i] + b[i];
}

__kernel void to_f16(__global const float* x, __global half* y, int n) {
    const size_t i = get_global_id(0);
    if (i < (size_t)n)
        y[i] = (half)x[i];
}

// C[i] = sum over the `nsplit` K-split partials P[s][i] (the skinny GEMMs' second pass), in split
// order — deterministic, unlike float atomics.
__kernel void sum_splits(__global const float* P, __global float* C, int n, int nsplit) {
    const int i = get_global_id(0);
    if (i >= n)
        return;
    float s = 0.0f;
    for (int k = 0; k < nsplit; k++)
        s += P[(size_t)k * n + i];
    C[i] = s;
}
