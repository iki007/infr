// The plumbing check: c = a + b.
__kernel void vadd(__global const float* a, __global const float* b, __global float* c, int n) {
    int i = get_global_id(0);
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
