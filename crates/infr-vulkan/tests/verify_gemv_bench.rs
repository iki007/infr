//! Speculative-verify GEMV bandwidth at the Qwen3.8-27B shapes (`docs/dflash.md`): the int8
//! multi-row GEMV (`mrow`) at m = 1, 4, 8, as weight GB/s. An m=8 verify should stream the weights
//! once, like m=1; the gap between the two columns is the verify's excess cost.
//! Run: INFR_DEV=Vulkan1 cargo test --release -p infr-vulkan --test verify_gemv_bench -- --ignored --nocapture
mod bench_support;

use infr_core::backend::{Backend, BufferUsage};
use infr_core::DType;

#[test]
#[ignore = "requires a Vulkan GPU (perf micro-bench)"]
fn verify_gemv_bench() {
    let be = bench_support::backend();
    let shapes = [
        (5120usize, 17408usize, "ffn gate/up"),
        (17408, 5120, "ffn down"),
        (5120, 10240, "dn qkv"),
        (6144, 5120, "attn out"),
    ];
    for dt in [DType::Q4K, DType::Q5K, DType::Q6K, DType::Iq4Xs] {
        for (k, n, label) in shapes {
            let wb = infr_testkit::synth_weight(dt, n * k, 42);
            let w = be.alloc(wb.len(), BufferUsage::Weights).unwrap();
            be.upload(w.as_ref(), &wb).unwrap();
            let x = be.alloc(8 * k * 4, BufferUsage::Activations).unwrap();
            be.upload(
                x.as_ref(),
                bytemuck::cast_slice(&infr_testkit::gen_f32(8 * k, 7)),
            )
            .unwrap();
            let qa = be.alloc(8 * k, BufferUsage::Activations).unwrap();
            let da = be.alloc(8 * k / 32 * 2, BufferUsage::Activations).unwrap();
            let sa = be.alloc(8 * k / 32 * 2, BufferUsage::Activations).unwrap();
            let y = be.alloc(8 * n * 4, BufferUsage::Activations).unwrap();
            let rec = be.recorder().unwrap();
            rec.quant_q8(x.as_ref(), qa.as_ref(), da.as_ref(), sa.as_ref(), 8, k);
            rec.finish().unwrap();
            let mut line = format!("{dt:?} {label} [{k}x{n}] {} MiB:", wb.len() >> 20);
            for m in [1usize, 4, 8] {
                let t = bench_support::time(&be, &format!("{dt:?} {label} m={m}"), 16, |rec| {
                    rec.linear_mmv_mrow(
                        dt,
                        w.as_ref(),
                        0,
                        qa.as_ref(),
                        da.as_ref(),
                        sa.as_ref(),
                        None,
                        y.as_ref(),
                        m,
                        k,
                        n,
                    )
                })
                .unwrap();
                if let Some(us) = t.mean_us() {
                    line += &format!(
                        "  m={m} {us:7.1} us {:4.0} GB/s",
                        wb.len() as f64 / (us * 1e-6) / 1e9
                    );
                }
            }
            println!("{line}");
        }
    }
}
