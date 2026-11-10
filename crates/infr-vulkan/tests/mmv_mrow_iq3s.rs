//! Host-reference parity for the IQ3_S int8 dp4a GEMV arm (`native_mmv_mrow.comp`'s `FMT_IQ3S`
//! `wdec`: shared-memory grid lookups, SWAR sign negation, one scale per 32-element sub-block).
//!
//! The reference dequantizes the weight with the host decoder (`infr_gguf::dequant_block`, which
//! re-derives IQ3_S from the GGUF layout element by element) and dots it in f64 against the GPU's
//! OWN `quant_q8` output (qa/dact read back), so only the weight-side math is under test. IQ3_S's
//! 110-byte blocks are 2-aligned only, so odd `out_f` puts both block parities on every shape, and
//! the shapes cover both mrow layouts on Xe2 (OUTS4 at in_f <= 5120, the 2-output one above) and
//! rows 1 (decode) and 3 (a verify batch).
//!
//! Run: cargo test -p infr-vulkan --test mmv_mrow_iq3s -- --ignored --nocapture
use infr_core::backend::{Backend, BufferUsage};
use infr_core::DType;
use infr_vulkan::VulkanBackend;

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn mmv_mrow_iq3s_matches_the_host_reference() {
    let be = VulkanBackend::new().unwrap();
    if !be.capabilities().i8_dot {
        eprintln!("skip: no int8 dot product");
        return;
    }
    let dt = DType::Iq3S;
    for (in_f, out_f) in [(1024usize, 37usize), (5120, 61), (6144, 9), (17408, 7)] {
        for rows in [1usize, 3] {
            let wbytes = infr_testkit::synth_weight(dt, in_f * out_f, (in_f + out_f) as u32);
            let wf = infr_gguf::dequant::dequant_block(dt, &wbytes).unwrap();
            let w = be.upload_weight_bytes(&wbytes).unwrap();
            let mut seed = (in_f * 31 + out_f) as u64;
            let xs: Vec<f32> = (0..rows * in_f).map(|_| lcg(&mut seed)).collect();
            let nblk = in_f / 32;
            let x = be.alloc(rows * in_f * 4, BufferUsage::Activations).unwrap();
            be.upload(x.as_ref(), bytemuck::cast_slice(&xs)).unwrap();
            let qa = be.alloc(rows * in_f, BufferUsage::Activations).unwrap();
            let dact = be.alloc(rows * nblk * 2, BufferUsage::Activations).unwrap();
            let sact = be.alloc(rows * nblk * 2, BufferUsage::Activations).unwrap();
            let y = be
                .alloc(rows * out_f * 4, BufferUsage::Activations)
                .unwrap();
            let rec = be.recorder().unwrap();
            rec.quant_q8(
                x.as_ref(),
                qa.as_ref(),
                dact.as_ref(),
                sact.as_ref(),
                rows,
                in_f,
            );
            rec.linear_mmv_mrow(
                dt,
                w.as_ref(),
                0,
                qa.as_ref(),
                dact.as_ref(),
                sact.as_ref(),
                None,
                y.as_ref(),
                rows,
                in_f,
                out_f,
            );
            rec.finish().unwrap();
            let mut qb = vec![0u8; rows * in_f];
            be.download(qa.as_ref(), &mut qb).unwrap();
            let mut db = vec![0u8; rows * nblk * 2];
            be.download(dact.as_ref(), &mut db).unwrap();
            let mut yb = vec![0u8; rows * out_f * 4];
            be.download(y.as_ref(), &mut yb).unwrap();
            let yg: &[f32] = bytemuck::cast_slice(&yb);
            let da: Vec<f64> = db
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&c| half::f16::from_le_bytes(c).to_f64())
                .collect();
            let mut worst = 0.0f64;
            for r in 0..rows {
                for o in 0..out_f {
                    let (mut acc, mut mag) = (0.0f64, 0.0f64);
                    for b in 0..nblk {
                        let (mut s, mut sm) = (0.0f64, 0.0f64);
                        for i in b * 32..b * 32 + 32 {
                            let p = wf[o * in_f + i] as f64 * (qb[r * in_f + i] as i8) as f64;
                            s += p;
                            sm += p.abs();
                        }
                        acc += da[r * nblk + b] * s;
                        mag += da[r * nblk + b].abs() * sm;
                    }
                    let err = (yg[r * out_f + o] as f64 - acc).abs() / mag.max(1e-30);
                    worst = worst.max(err);
                }
            }
            assert!(
                worst < 1e-5,
                "{in_f}x{out_f} rows={rows}: worst scaled error {worst:e} vs the host reference"
            );
            println!("{in_f}x{out_f} rows={rows}: worst scaled error {worst:e}");
        }
    }
}
