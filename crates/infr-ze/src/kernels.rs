//! The `ocloc`-built kernel binaries (see `build.rs`). Absent (`None`) when the crate was built
//! without `ocloc`.

macro_rules! bin {
    ($name:literal) => {{
        #[cfg(infr_ze_kernels)]
        {
            Some(&include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".bin"))[..])
        }
        #[cfg(not(infr_ze_kernels))]
        {
            None::<&'static [u8]>
        }
    }};
}

/// `kernels/vadd.cl`.
pub fn vadd() -> Option<&'static [u8]> {
    bin!("vadd")
}

/// `kernels/gemm_f16.cl`: the DPAS + 2D-block-load f16 GEMM probe.
pub fn gemm_f16() -> Option<&'static [u8]> {
    bin!("gemm_f16")
}

/// `kernels/gemm_kquant.cl`: `C = A · Wᵀ` with W in a GGUF K-quant, dequantized in registers into
/// DPAS's B operand — kernels `gemm_q4k`, `gemm_q5k`.
pub fn gemm_kquant() -> Option<&'static [u8]> {
    bin!("gemm_kquant")
}

/// `kernels/gemm_iq4.cl`: `C = A · Wᵀ` with W in GGUF IQ4_XS (kernel `gemm_iq4xs`), the codebook
/// through a shared-memory byte-pair table.
pub fn gemm_iq4() -> Option<&'static [u8]> {
    bin!("gemm_iq4")
}

/// `kernels/elementwise.cl`: the ops between the prefill GEMMs — `rmsnorm`, `swiglu`, `add`,
/// `to_f16`.
pub fn elementwise() -> Option<&'static [u8]> {
    bin!("elementwise")
}

/// `kernels/gemm_kq16.cl`: the K-quants with one scale per 16 values and no min — kernels
/// `gemm_q6k`, `gemm_q3k` (K % 512 == 0: superblock pairs).
pub fn gemm_kq16() -> Option<&'static [u8]> {
    bin!("gemm_kq16")
}

/// `kernels/attn_fa.cl`: DPAS flash-attention prefill (`attn_fa256`: head dim 256).
pub fn attn_fa() -> Option<&'static [u8]> {
    bin!("attn_fa")
}
