//! `Recorder::rmsnorm_q8` (the decode norm fused with the next int8 GEMV's activation
//! quantization, `rmsnorm_q8.comp`) against the two dispatches it replaces, `rmsnorm` +
//! `quant_q8`:
//!
//! * its int8 output is EXACTLY `quant_q8` of its own f32 output (same values through the same
//!   math), so the int8 GEMV reads what the unfused path would have produced from that row;
//! * its f32 output matches `rmsnorm`'s to rounding (the sum of squares is reduced in a different
//!   order, so the two scales may differ in the last bits).
//!
//! Run: cargo test -p infr-vulkan --test rmsnorm_q8_parity -- --ignored --nocapture
use infr_core::backend::{Backend, BufferUsage};
use infr_vulkan::VulkanBackend;

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn rmsnorm_q8_matches_rmsnorm_then_quant_q8() {
    let be = VulkanBackend::new().unwrap();
    let eps = 1e-6f32;
    for dim in [2048usize, 2560, 4096, 5120] {
        let mut seed = dim as u64;
        // Activations with a few large outliers per block, like a residual stream.
        let xs: Vec<f32> = (0..dim)
            .map(|i| lcg(&mut seed) * if i % 37 == 0 { 20.0 } else { 1.0 })
            .collect();
        let ws: Vec<f32> = (0..dim).map(|_| 0.5 + lcg(&mut seed).abs()).collect();
        let nblk = dim / 32;
        let x = be.alloc(dim * 4, BufferUsage::Activations).unwrap();
        let w = be.alloc(dim * 4, BufferUsage::Activations).unwrap();
        be.upload(x.as_ref(), bytemuck::cast_slice(&xs)).unwrap();
        be.upload(w.as_ref(), bytemuck::cast_slice(&ws)).unwrap();
        let buf = |n: usize| be.alloc(n, BufferUsage::Activations).unwrap();
        let (y_ref, qa_ref, d_ref, s_ref) = (buf(dim * 4), buf(dim), buf(nblk * 2), buf(nblk * 2));
        let (y, qa, d, s) = (buf(dim * 4), buf(dim), buf(nblk * 2), buf(nblk * 2));
        let (qa_re, d_re, s_re) = (buf(dim), buf(nblk * 2), buf(nblk * 2));

        let rec = be.recorder().unwrap();
        rec.rmsnorm(x.as_ref(), w.as_ref(), y_ref.as_ref(), 1, dim, eps);
        rec.quant_q8(
            y_ref.as_ref(),
            qa_ref.as_ref(),
            d_ref.as_ref(),
            s_ref.as_ref(),
            1,
            dim,
        );
        rec.rmsnorm_q8(
            x.as_ref(),
            w.as_ref(),
            y.as_ref(),
            qa.as_ref(),
            d.as_ref(),
            s.as_ref(),
            dim,
            eps,
        );
        // Requantize the fused kernel's own f32 output with the standalone kernel.
        rec.quant_q8(
            y.as_ref(),
            qa_re.as_ref(),
            d_re.as_ref(),
            s_re.as_ref(),
            1,
            dim,
        );
        rec.finish().unwrap();

        let get = |b: &dyn infr_core::backend::Buffer, n: usize| {
            let mut v = vec![0u8; n];
            be.download(b, &mut v).unwrap();
            v
        };
        assert_eq!(
            get(qa.as_ref(), dim),
            get(qa_re.as_ref(), dim),
            "dim {dim}: qa"
        );
        assert_eq!(
            get(d.as_ref(), nblk * 2),
            get(d_re.as_ref(), nblk * 2),
            "dim {dim}: dact"
        );
        assert_eq!(
            get(s.as_ref(), nblk * 2),
            get(s_re.as_ref(), nblk * 2),
            "dim {dim}: sact"
        );

        let yv: Vec<f32> = bytemuck::cast_slice(&get(y.as_ref(), dim * 4)).to_vec();
        let yr: Vec<f32> = bytemuck::cast_slice(&get(y_ref.as_ref(), dim * 4)).to_vec();
        let worst = yv
            .iter()
            .zip(&yr)
            .map(|(a, b)| (a - b).abs() / b.abs().max(1e-3))
            .fold(0.0f32, f32::max);
        assert!(
            worst < 1e-5,
            "dim {dim}: f32 output rel err {worst:e} vs rmsnorm"
        );
        // The int8 output may differ from the unfused path's only where a last-bit difference in y
        // crosses a rounding boundary: by one step, and rarely.
        let qv = get(qa.as_ref(), dim);
        let qr = get(qa_ref.as_ref(), dim);
        let mut off = 0usize;
        for (a, b) in qv.iter().zip(&qr) {
            let diff = (*a as i8 as i32 - *b as i8 as i32).abs();
            assert!(diff <= 1, "dim {dim}: int8 step {diff} vs rmsnorm+quant_q8");
            off += (diff != 0) as usize;
        }
        assert!(
            off * 100 <= dim,
            "dim {dim}: {off}/{dim} int8 values off by one"
        );
        println!("dim {dim}: f32 rel err {worst:e}, int8 off-by-one {off}/{dim}");
    }
}
