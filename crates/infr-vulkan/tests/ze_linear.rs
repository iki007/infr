//! The Level Zero GEMM offload (`INFR_ZE_GEMM`, `src/ze.rs`) end to end through the adapter: a
//! prefill `Linear` into an internal tensor followed by a residual `Add` (the pair the Vulkan
//! path fuses), run with the offload on and off, against the host reference.
//!
//! Hardware-gated (`#[ignore]`): an Intel Xe2 card driven by both Mesa ANV and Level Zero.
//! `INFR_ZE_VK` picks the Vulkan device index (default 1, the B70 box's headless card).
#![cfg(target_os = "linux")]

use infr_core::config::Config;
use infr_core::graph::{Graph, Op};
use infr_core::tensor::{DType, TensorDesc};
use infr_testkit::{
    case_inputs, dequant_oracle, f32_bytes, gen_f32, ref_linear, run_graph, LinearCase,
};
use infr_vulkan::VulkanBackend;
use std::sync::Arc;

fn backend(ze: bool) -> Option<VulkanBackend> {
    let idx = std::env::var("INFR_ZE_VK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let mut cfg = Config::default();
    cfg.kernels.vulkan.ze_gemm = ze;
    VulkanBackend::new_on_with(idx, Arc::new(cfg)).ok()
}

fn run(be_: &VulkanBackend, case: LinearCase, resid: &[f32]) -> Vec<f32> {
    let LinearCase {
        dtype,
        m,
        in_f,
        out_f,
        ..
    } = case;
    let (w_bytes, x) = case_inputs(case);
    let mut g = Graph::new();
    let xi = g.input(TensorDesc::new(vec![m, in_f], DType::F32));
    let wi = g.weight(TensorDesc::new(vec![out_f, in_f], dtype));
    let ri = g.input(TensorDesc::new(vec![m, out_f], DType::F32));
    let t = g.internal(TensorDesc::new(vec![m, out_f], DType::F32));
    let yi = g.output(TensorDesc::new(vec![m, out_f], DType::F32));
    g.push(Op::Linear {
        x: xi,
        weight: wi,
        dst: t,
        m: m as u32,
        in_f: in_f as u32,
        out_f: out_f as u32,
        w_off: 0,
    });
    g.push(Op::Add {
        a: t,
        b: ri,
        dst: yi,
        n: (m * out_f) as u32,
    });
    run_graph(
        be_,
        &g,
        &[(xi, f32_bytes(&x)), (wi, w_bytes), (ri, f32_bytes(resid))],
        yi,
        m * out_f,
    )
}

#[test]
#[ignore = "needs an Intel Xe2 GPU with both Mesa ANV and Level Zero"]
fn ze_linear_add_matches_vulkan_and_host() {
    let (Some(vk), Some(ze)) = (backend(false), backend(true)) else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    for dt in [DType::Q4K, DType::Q5K, DType::Iq4Xs] {
        for (m, in_f, out_f) in [
            (512usize, 1024usize, 512usize),
            (300, 2048, 1024),
            (512, 2560, 18432),
            (353, 2560, 18432),
        ] {
            let case = LinearCase::new(dt, m, in_f, out_f).with_seed(0x77);
            let resid = gen_f32(m * out_f, 0x5151);
            let (w_bytes, x) = case_inputs(case);
            let w_ref = dequant_oracle(dt, &w_bytes);
            let want: Vec<f32> = ref_linear(&x, &w_ref, m, in_f, out_f)
                .iter()
                .zip(&resid)
                .map(|(a, b)| a + b)
                .collect();
            let got_vk = run(&vk, case, &resid);
            let got_ze = run(&ze, case, &resid);
            let mag = want.iter().fold(0f32, |a, v| a.max(v.abs()));
            let err = |got: &[f32]| {
                got.iter()
                    .zip(&want)
                    .fold(0f32, |a, (g, w)| a.max((g - w).abs()))
            };
            let (e_vk, e_ze) = (err(&got_vk), err(&got_ze));
            let first_bad = got_ze
                .iter()
                .zip(&want)
                .position(|(g, w)| (g - w).abs() > 2e-2 * mag.max(1.0));
            println!(
                "{dt:?} m={m} {in_f}->{out_f}: max|C| {mag:.2}, err vulkan {e_vk:.3e}, \
                 level zero {e_ze:.3e}, first bad {first_bad:?}"
            );
            assert!(
                first_bad.is_none(),
                "{dt:?} m={m} {in_f}->{out_f}: Level Zero offload mismatch at {first_bad:?}: \
                 got {} want {}",
                got_ze[first_bad.unwrap()],
                want[first_bad.unwrap()]
            );
        }
    }
}
