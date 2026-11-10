//! Level Zero probes on the Intel GPU. `#[ignore]`d: they need a Level Zero GPU and the
//! `ocloc`-built kernels. `INFR_ZE_PCI=03:00.0` picks the card (default: the first GPU).
//!
//! Run: cargo test --release -p infr-ze --test probe -- --ignored --nocapture
use infr_ze::{Gpu, Ze};

fn gpu() -> Option<Gpu> {
    let ze = match Ze::open() {
        Ok(z) => z,
        Err(e) => {
            eprintln!("skip: {e}");
            return None;
        }
    };
    for d in ze.devices().ok()? {
        eprintln!(
            "ze device {} [{}] {} EUs @ {} MHz",
            d.name, d.pci, d.eus, d.clock_mhz
        );
    }
    let pci = std::env::var("INFR_ZE_PCI").ok();
    Gpu::open(&ze, pci.as_deref()).ok()
}

#[test]
#[ignore = "requires a Level Zero GPU"]
fn vadd_matches_host() {
    let (Some(g), Some(bin)) = (gpu(), infr_ze::kernels::vadd()) else {
        eprintln!("skip: no Level Zero GPU or no ocloc kernels");
        return;
    };
    let n = 1 << 20;
    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
    let b: Vec<f32> = (0..n).map(|i| 1.0 - i as f32).collect();
    let (da, db, dc) = (
        g.alloc(n * 4).unwrap(),
        g.alloc(n * 4).unwrap(),
        g.alloc(n * 4).unwrap(),
    );
    g.upload(&da, bytemuck_f32(&a)).unwrap();
    g.upload(&db, bytemuck_f32(&b)).unwrap();
    let m = g.module(bin).unwrap();
    let k = m.kernel("vadd").unwrap();
    k.group_size(256, 1, 1).unwrap();
    k.arg_buf(0, &da).unwrap();
    k.arg_buf(1, &db).unwrap();
    k.arg_buf(2, &dc).unwrap();
    k.arg(3, n as i32).unwrap();
    g.launch(&k, [(n / 256) as u32, 1, 1]).unwrap();
    let mut out = vec![0u8; n * 4];
    g.download(&dc, &mut out).unwrap();
    let c: Vec<f32> = out
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| f32::from_le_bytes(*w))
        .collect();
    for i in 0..n {
        assert_eq!(c[i], a[i] + b[i], "element {i}");
    }
    let t = g.time_launches(&k, [(n / 256) as u32, 1, 1], 20).unwrap();
    let best = t.iter().cloned().fold(f64::MAX, f64::min);
    println!(
        "vadd {n}: {:.1} us ({:.0} GB/s)",
        best / 1e3,
        (12 * n) as f64 / best
    );
}

fn bytemuck_f32(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding or invalid bit patterns as bytes.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

fn f16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| half::f16::from_f32(x).to_le_bytes())
        .collect()
}

/// The DPAS GEMM probe: `C = A · Wᵀ` against a host f32 reference at a small shape, then timed at
/// the Qwen3.8-27B gate/up prefill shape (m=512, 5120 -> 17408). INFR_ZE_GEMM="M,N,K" overrides
/// the timed shape.
#[test]
#[ignore = "requires a Level Zero GPU"]
fn gemm_f16_probe() {
    let (Some(g), Some(bin)) = (gpu(), infr_ze::kernels::gemm_f16()) else {
        eprintln!("skip: no Level Zero GPU or no ocloc kernels");
        return;
    };
    let module = g.module(bin).unwrap();
    let k = module.kernel("gemm_f16").unwrap();
    k.group_size(128, 1, 1).unwrap();
    let run = |m: usize,
               n: usize,
               kk: usize,
               a: &infr_ze::DevBuf,
               w: &infr_ze::DevBuf,
               c: &infr_ze::DevBuf| {
        k.arg_buf(0, a).unwrap();
        k.arg_buf(1, w).unwrap();
        k.arg_buf(2, c).unwrap();
        k.arg(3, m as i32).unwrap();
        k.arg(4, n as i32).unwrap();
        k.arg(5, kk as i32).unwrap();
        [(m / 128) as u32, (n / 128) as u32, 1]
    };
    // correctness
    let (m, n, kk) = (128usize, 256usize, 256usize);
    let mut seed = 7u64;
    let a: Vec<f32> = (0..m * kk)
        .map(|_| half::f16::from_f32(lcg(&mut seed)).to_f32())
        .collect();
    let w: Vec<f32> = (0..n * kk)
        .map(|_| half::f16::from_f32(lcg(&mut seed)).to_f32())
        .collect();
    let (da, dw, dc) = (
        g.alloc(m * kk * 2).unwrap(),
        g.alloc(n * kk * 2).unwrap(),
        g.alloc(m * n * 4).unwrap(),
    );
    g.upload(&da, &f16_bytes(&a)).unwrap();
    g.upload(&dw, &f16_bytes(&w)).unwrap();
    let groups = run(m, n, kk, &da, &dw, &dc);
    g.launch(&k, groups).unwrap();
    let mut out = vec![0u8; m * n * 4];
    g.download(&dc, &mut out).unwrap();
    let c: Vec<f32> = out
        .as_chunks::<4>()
        .0
        .iter()
        .map(|x| f32::from_le_bytes(*x))
        .collect();
    let mut worst = 0.0f64;
    for r in 0..m {
        for col in 0..n {
            let mut s = 0.0f64;
            for i in 0..kk {
                s += a[r * kk + i] as f64 * w[col * kk + i] as f64;
            }
            worst = worst.max((c[r * n + col] as f64 - s).abs());
        }
    }
    println!("gemm_f16 {m}x{n}x{kk}: worst abs err {worst:e}");
    assert!(worst < 1e-2, "gemm_f16 mismatch {worst}");
    // timing
    let (m, n, kk) = std::env::var("INFR_ZE_GEMM")
        .ok()
        .map(|s| {
            let v: Vec<usize> = s.split(',').map(|x| x.parse().unwrap()).collect();
            (v[0], v[1], v[2])
        })
        .unwrap_or((512, 17408, 5120));
    let (da, dw, dc) = (
        g.alloc(m * kk * 2).unwrap(),
        g.alloc(n * kk * 2).unwrap(),
        g.alloc(m * n * 4).unwrap(),
    );
    let mut seed = 9u64;
    let a: Vec<f32> = (0..m * kk).map(|_| lcg(&mut seed)).collect();
    let w: Vec<f32> = (0..n * kk).map(|_| lcg(&mut seed)).collect();
    g.upload(&da, &f16_bytes(&a)).unwrap();
    g.upload(&dw, &f16_bytes(&w)).unwrap();
    let groups = run(m, n, kk, &da, &dw, &dc);
    for reps in [10u32, 200] {
        let t = g.time_launches(&k, groups, reps).unwrap();
        let mut s = t.clone();
        s.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let med = s[s.len() / 2];
        let tf = 2.0 * (m * n * kk) as f64 / med / 1e3;
        println!(
            "gemm_f16 m={m} {kk}->{n} x{reps}: median {:.3} ms, {tf:.1} TFLOPS",
            med / 1e6
        );
    }
}
