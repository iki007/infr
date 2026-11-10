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

/// `kernels/gemm_q4k.cl`: `C = A · Wᵀ` with W in GGUF Q4_K, dequantized in registers into DPAS's
/// B operand.
pub fn gemm_q4k() -> Option<&'static [u8]> {
    bin!("gemm_q4k")
}
