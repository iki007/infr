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

/// A GGUF weight format for the quantized GEMM probes: superblock size and bytes, the byte offsets
/// of its f16 scale fields (set to small sane values; every other byte is random, which is a valid
/// block for these formats), and the kernel's workgroup tile.
struct QFormat {
    name: &'static str,
    dtype: infr_core::DType,
    block_bytes: usize,
    /// Values per block (256 for the superblock formats, 32 for IQ4_NL).
    block_vals: usize,
    f16_fields: &'static [usize],
    wg_m: usize,
    wg_n: usize,
}

fn random_quant(fmt: &QFormat, n: usize, k: usize, seed: &mut u64) -> Vec<u8> {
    let nb = k / fmt.block_vals;
    let mut w = vec![0u8; n * nb * fmt.block_bytes];
    for (i, b) in w.iter_mut().enumerate() {
        *b = ((lcg(seed) + 1.0) * 127.9) as u8 ^ (i as u8);
    }
    for blk in w.chunks_exact_mut(fmt.block_bytes) {
        for &off in fmt.f16_fields {
            let v = 0.0005 + 0.002 * (lcg(seed) + 1.0);
            blk[off..off + 2].copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
        }
    }
    w
}

/// Correctness against the host dequant (`infr_gguf::dequant`) at a small shape, then timing at the
/// 27B gate/up prefill shape (m=512, 5120 -> 17408; INFR_ZE_GEMM="M,N,K" overrides).
fn quant_gemm_probe(fmt: &QFormat, bin: &[u8]) {
    let Some(g) = gpu() else {
        eprintln!("skip: no Level Zero GPU");
        return;
    };
    let module = g.module(bin).unwrap();
    let kern = module.kernel(fmt.name).unwrap();
    kern.group_size(128, 1, 1).unwrap();
    let launch_dims = |m: usize, n: usize| [m.div_ceil(fmt.wg_m) as u32, (n / fmt.wg_n) as u32, 1];
    let set = |m: usize,
               n: usize,
               k: usize,
               a: &infr_ze::DevBuf,
               w: &infr_ze::DevBuf,
               c: &infr_ze::DevBuf| {
        kern.arg_buf(0, a).unwrap();
        kern.arg_buf(1, w).unwrap();
        kern.arg_buf(2, c).unwrap();
        kern.arg(3, m as i32).unwrap();
        kern.arg(4, n as i32).unwrap();
        kern.arg(5, k as i32).unwrap();
    };
    // correctness; m not a multiple of the tile, to cover the bounds-checked rows
    let (m, n, k) = (200usize, 128usize, 1024usize);
    let mut seed = 11u64;
    let a: Vec<f32> = (0..m * k)
        .map(|_| half::f16::from_f32(lcg(&mut seed)).to_f32())
        .collect();
    let wq = random_quant(fmt, n, k, &mut seed);
    let row_bytes = k / fmt.block_vals * fmt.block_bytes;
    let wf: Vec<f32> = wq
        .chunks_exact(row_bytes)
        .flat_map(|r| infr_gguf::dequant::dequant_block(fmt.dtype, r).unwrap())
        .collect();
    let (da, dw, dc) = (
        g.alloc(m * k * 2).unwrap(),
        g.alloc(wq.len()).unwrap(),
        g.alloc(m * n * 4).unwrap(),
    );
    g.upload(&da, &f16_bytes(&a)).unwrap();
    g.upload(&dw, &wq).unwrap();
    set(m, n, k, &da, &dw, &dc);
    g.launch(&kern, launch_dims(m, n)).unwrap();
    let mut out = vec![0u8; m * n * 4];
    g.download(&dc, &mut out).unwrap();
    let c: Vec<f32> = out
        .as_chunks::<4>()
        .0
        .iter()
        .map(|x| f32::from_le_bytes(*x))
        .collect();
    let (mut worst, mut big) = (0.0f64, 0.0f64);
    for r in 0..m {
        for col in 0..n {
            let mut s = 0.0f64;
            for i in 0..k {
                s += a[r * k + i] as f64 * wf[col * k + i] as f64;
            }
            worst = worst.max((c[r * n + col] as f64 - s).abs());
            big = big.max(s.abs());
        }
    }
    println!(
        "{} {m}x{n}x{k}: worst abs err {worst:.3e} (max |C| {big:.2})",
        fmt.name
    );
    // INFR_ZE_NOCHECK: time a deliberately wrong kernel variant (e.g. a bound with no dequant).
    if std::env::var_os("INFR_ZE_NOCHECK").is_none() {
        assert!(worst < 2e-3 * big, "{} mismatch {worst}", fmt.name);
    }
    // timing
    let (m, n, k) = std::env::var("INFR_ZE_GEMM")
        .ok()
        .map(|s| {
            let v: Vec<usize> = s.split(',').map(|x| x.parse().unwrap()).collect();
            (v[0], v[1], v[2])
        })
        .unwrap_or((512, 17408, 5120));
    let a: Vec<f32> = (0..m * k).map(|_| lcg(&mut seed)).collect();
    let wq = random_quant(fmt, n, k, &mut seed);
    let (da, dw, dc) = (
        g.alloc(m * k * 2).unwrap(),
        g.alloc(wq.len()).unwrap(),
        g.alloc(m * n * 4).unwrap(),
    );
    g.upload(&da, &f16_bytes(&a)).unwrap();
    g.upload(&dw, &wq).unwrap();
    set(m, n, k, &da, &dw, &dc);
    for reps in [10u32, 200] {
        let t = g.time_launches(&kern, launch_dims(m, n), reps).unwrap();
        let mut s = t.clone();
        s.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let med = s[s.len() / 2];
        let tf = 2.0 * (m * n * k) as f64 / med / 1e3;
        println!(
            "{} m={m} {k}->{n} x{reps}: median {:.3} ms, {tf:.1} TFLOPS",
            fmt.name,
            med / 1e6
        );
    }
}

#[test]
#[ignore = "requires a Level Zero GPU"]
fn gemm_q4k_probe() {
    let Some(bin) = infr_ze::kernels::gemm_kquant() else {
        eprintln!("skip: no ocloc kernels");
        return;
    };
    let fmt = QFormat {
        name: "gemm_q4k",
        dtype: infr_core::DType::Q4K,
        block_bytes: 144,
        block_vals: 256,
        f16_fields: &[0, 2],
        wg_m: 256,
        wg_n: 64,
    };
    quant_gemm_probe(&fmt, bin);
}

#[test]
#[ignore = "requires a Level Zero GPU"]
fn gemm_q5k_probe() {
    let Some(bin) = infr_ze::kernels::gemm_kquant() else {
        eprintln!("skip: no ocloc kernels");
        return;
    };
    let fmt = QFormat {
        name: "gemm_q5k",
        dtype: infr_core::DType::Q5K,
        block_bytes: 176,
        block_vals: 256,
        f16_fields: &[0, 2],
        wg_m: 256,
        wg_n: 64,
    };
    quant_gemm_probe(&fmt, bin);
}

#[test]
#[ignore = "requires a Level Zero GPU"]
fn gemm_iq4xs_probe() {
    let Some(bin) = infr_ze::kernels::gemm_iq4() else {
        eprintln!("skip: no ocloc kernels");
        return;
    };
    let fmt = QFormat {
        name: "gemm_iq4xs",
        dtype: infr_core::DType::Iq4Xs,
        block_bytes: 136,
        block_vals: 256,
        f16_fields: &[0],
        wg_m: 256,
        wg_n: 64,
    };
    quant_gemm_probe(&fmt, bin);
}

#[test]
#[ignore = "requires a Level Zero GPU"]
fn gemm_q6k_probe() {
    let Some(bin) = infr_ze::kernels::gemm_kq16() else {
        eprintln!("skip: no ocloc kernels");
        return;
    };
    let fmt = QFormat {
        name: "gemm_q6k",
        dtype: infr_core::DType::Q6K,
        block_bytes: 210,
        block_vals: 256,
        f16_fields: &[208],
        wg_m: 256,
        wg_n: 64,
    };
    quant_gemm_probe(&fmt, bin);
}

#[test]
#[ignore = "requires a Level Zero GPU"]
fn gemm_q3k_probe() {
    let Some(bin) = infr_ze::kernels::gemm_kq16() else {
        eprintln!("skip: no ocloc kernels");
        return;
    };
    let fmt = QFormat {
        name: "gemm_q3k",
        dtype: infr_core::DType::Q3K,
        block_bytes: 110,
        block_vals: 256,
        f16_fields: &[108],
        wg_m: 256,
        wg_n: 64,
    };
    quant_gemm_probe(&fmt, bin);
}

#[test]
#[ignore = "requires a Level Zero GPU"]
fn gemm_iq4nl_probe() {
    let Some(bin) = infr_ze::kernels::gemm_iq4() else {
        eprintln!("skip: no ocloc kernels");
        return;
    };
    let fmt = QFormat {
        name: "gemm_iq4nl",
        dtype: infr_core::DType::Iq4Nl,
        block_bytes: 18,
        block_vals: 32,
        f16_fields: &[0],
        wg_m: 256,
        wg_n: 64,
    };
    quant_gemm_probe(&fmt, bin);
}

/// Host attention (f64), the `attn_fa256` semantics: causal from `pos`, optional window, GQA.
#[allow(clippy::too_many_arguments)]
fn host_attn(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    q_len: usize,
    kv_len: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    pos: usize,
    window: usize,
) -> Vec<f32> {
    let scale = 1.0 / (hd as f64).sqrt();
    let mut o = vec![0f32; q_len * nh * hd];
    for r in 0..q_len {
        let qp = pos + r;
        for h in 0..nh {
            let kvh = h / (nh / nkv);
            let lo = if window > 0 && qp + 1 > window {
                qp + 1 - window
            } else {
                0
            };
            let hi = (qp + 1).min(kv_len);
            let qrow = &q[(r * nh + h) * hd..][..hd];
            let sc: Vec<f64> = (lo..hi)
                .map(|j| {
                    let krow = &k[(j * nkv + kvh) * hd..][..hd];
                    qrow.iter()
                        .zip(krow)
                        .map(|(a, b)| *a as f64 * *b as f64)
                        .sum::<f64>()
                        * scale
                })
                .collect();
            let mx = sc.iter().cloned().fold(f64::MIN, f64::max);
            let w: Vec<f64> = sc.iter().map(|s| (s - mx).exp()).collect();
            let sum: f64 = w.iter().sum();
            for d in 0..hd {
                let mut acc = 0f64;
                for (i, j) in (lo..hi).enumerate() {
                    acc += w[i] * v[(j * nkv + kvh) * hd + d] as f64;
                }
                o[(r * nh + h) * hd + d] = if sum > 0.0 { (acc / sum) as f32 } else { 0.0 };
            }
        }
    }
    o
}

#[test]
#[ignore = "requires a Level Zero GPU"]
fn attn_fa256_probe() {
    let (Some(g), Some(bin)) = (gpu(), infr_ze::kernels::attn_fa()) else {
        eprintln!("skip: no Level Zero GPU or no ocloc kernels");
        return;
    };
    let module = g.module(bin).unwrap();
    let kern = module.kernel("attn_fa256").unwrap();
    kern.group_size(128, 1, 1).unwrap();
    let hd = 256;
    let run = |q_len: usize,
               kv_len: usize,
               nh: usize,
               nkv: usize,
               pos: usize,
               window: usize,
               dq: &infr_ze::DevBuf,
               dk: &infr_ze::DevBuf,
               dv: &infr_ze::DevBuf,
               dout: &infr_ze::DevBuf| {
        kern.arg_buf(0, dq).unwrap();
        kern.arg_buf(1, dk).unwrap();
        kern.arg_buf(2, dv).unwrap();
        kern.arg_buf(3, dout).unwrap();
        for (i, v) in [q_len, kv_len, nh, nkv, pos, window].iter().enumerate() {
            kern.arg(4 + i as u32, *v as i32).unwrap();
        }
        kern.arg(10, 0.0f32).unwrap();
        [(q_len.div_ceil(32) * nh) as u32, 1, 1]
    };
    for (q_len, kv_len, nh, nkv, window) in [
        (100usize, 300usize, 6usize, 2usize, 0usize),
        (130, 260, 4, 4, 70),
    ] {
        let pos = kv_len - q_len;
        let mut seed = 5u64;
        let f16r = |n: usize, seed: &mut u64| -> Vec<f32> {
            (0..n)
                .map(|_| half::f16::from_f32(lcg(seed)).to_f32())
                .collect()
        };
        let q = f16r(q_len * nh * hd, &mut seed);
        let k = f16r(kv_len * nkv * hd, &mut seed);
        let v = f16r(kv_len * nkv * hd, &mut seed);
        let (dq, dk, dv, dout) = (
            g.alloc(q.len() * 2).unwrap(),
            g.alloc(k.len() * 2).unwrap(),
            g.alloc(v.len() * 2).unwrap(),
            g.alloc(q_len * nh * hd * 4).unwrap(),
        );
        g.upload(&dq, &f16_bytes(&q)).unwrap();
        g.upload(&dk, &f16_bytes(&k)).unwrap();
        g.upload(&dv, &f16_bytes(&v)).unwrap();
        let groups = run(q_len, kv_len, nh, nkv, pos, window, &dq, &dk, &dv, &dout);
        g.launch(&kern, groups).unwrap();
        let mut out = vec![0u8; q_len * nh * hd * 4];
        g.download(&dout, &mut out).unwrap();
        let got: Vec<f32> = out
            .as_chunks::<4>()
            .0
            .iter()
            .map(|x| f32::from_le_bytes(*x))
            .collect();
        let want = host_attn(&q, &k, &v, q_len, kv_len, nh, nkv, hd, pos, window);
        assert!(
            got.iter().all(|x| x.is_finite()),
            "non-finite attention output"
        );
        let worst = got
            .iter()
            .zip(&want)
            .fold(0f32, |a, (g, w)| a.max((g - w).abs()));
        let mag = want.iter().fold(0f32, |a, w| a.max(w.abs()));
        println!("attn_fa256 q={q_len} kv={kv_len} nh={nh} nkv={nkv} window={window}: worst abs err {worst:.3e} (max |O| {mag:.3})");
        assert!(worst < 1e-2 * mag.max(0.1), "attn mismatch {worst}");
    }
    // timing: the 27B's pp512 at 32k depth
    let (q_len, kv_len, nh, nkv) = (512usize, 33280usize, 24usize, 4usize);
    let (dq, dk, dv, dout) = (
        g.alloc(q_len * nh * hd * 2).unwrap(),
        g.alloc(kv_len * nkv * hd * 2).unwrap(),
        g.alloc(kv_len * nkv * hd * 2).unwrap(),
        g.alloc(q_len * nh * hd * 4).unwrap(),
    );
    let mut seed = 9u64;
    g.upload(
        &dq,
        &f16_bytes(
            &(0..q_len * nh * hd)
                .map(|_| lcg(&mut seed))
                .collect::<Vec<_>>(),
        ),
    )
    .unwrap();
    let kv: Vec<f32> = (0..kv_len * nkv * hd).map(|_| lcg(&mut seed)).collect();
    g.upload(&dk, &f16_bytes(&kv)).unwrap();
    g.upload(&dv, &f16_bytes(&kv)).unwrap();
    let groups = run(
        q_len,
        kv_len,
        nh,
        nkv,
        kv_len - q_len,
        0,
        &dq,
        &dk,
        &dv,
        &dout,
    );
    let t = g.time_launches(&kern, groups, 5).unwrap();
    let mut s = t.clone();
    s.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let med = s[s.len() / 2];
    let flops = 4.0 * q_len as f64 * kv_len as f64 * hd as f64 * nh as f64;
    println!(
        "attn_fa256 q={q_len} kv={kv_len} nh={nh} nkv={nkv}: median {:.2} ms, {:.1} TFLOPS (Vulkan attn_xmx_fa: 69 ms)",
        med / 1e6,
        flops / med / 1e3
    );
}

/// The skinny (M <= 16) K-quant GEMMs (`gemm_q4k_m8` / `_m16`, `gemm_q5k_*` + `sum_splits`): host
/// reference at a small shape with and without split-K, then the verify shapes of Qwen3.8-27B at
/// m = 8 (kernel time, split-sum included, as weight GB/s).
fn skinny_probe(fmt: &QFormat, bin: Option<&[u8]>, m8: &str, m16: &str) {
    let (Some(g), Some(bin), Some(ew)) = (gpu(), bin, infr_ze::kernels::elementwise()) else {
        eprintln!("skip: no Level Zero GPU or no ocloc kernels");
        return;
    };
    let module = g.module(bin).unwrap();
    let ewm = g.module(ew).unwrap();
    let sum = ewm.kernel("sum_splits").unwrap();
    sum.group_size(256, 1, 1).unwrap();
    let kern = |name: &str| {
        let k = module.kernel(name).unwrap();
        k.group_size(128, 1, 1).unwrap();
        k
    };
    let (k8, k16) = (kern(m8), kern(m16));
    let mut seed = 5u64;
    // correctness
    let (n, k) = (512usize, 1024usize);
    let wq = random_quant(fmt, n, k, &mut seed);
    let row_bytes = k / fmt.block_vals * fmt.block_bytes;
    let wf: Vec<f32> = wq
        .chunks_exact(row_bytes)
        .flat_map(|r| infr_gguf::dequant::dequant_block(fmt.dtype, r).unwrap())
        .collect();
    let dw = g.alloc(wq.len()).unwrap();
    g.upload(&dw, &wq).unwrap();
    for (m, nsplit) in [(8usize, 1usize), (5, 3), (13, 1), (16, 4)] {
        let kk = if m <= 8 { &k8 } else { &k16 };
        let a: Vec<f32> = (0..m * k)
            .map(|_| half::f16::from_f32(lcg(&mut seed)).to_f32())
            .collect();
        let (da, dp, dc) = (
            g.alloc(m * k * 2).unwrap(),
            g.alloc(nsplit * m * n * 4).unwrap(),
            g.alloc(m * n * 4).unwrap(),
        );
        g.upload(&da, &f16_bytes(&a)).unwrap();
        kk.arg_buf(0, &da).unwrap();
        kk.arg_buf(1, &dw).unwrap();
        kk.arg_buf(2, if nsplit > 1 { &dp } else { &dc }).unwrap();
        kk.arg(3, m as i32).unwrap();
        kk.arg(4, n as i32).unwrap();
        kk.arg(5, k as i32).unwrap();
        kk.arg(6, nsplit as i32).unwrap();
        g.launch(kk, [n.div_ceil(256) as u32, nsplit as u32, 1])
            .unwrap();
        if nsplit > 1 {
            sum.arg_buf(0, &dp).unwrap();
            sum.arg_buf(1, &dc).unwrap();
            sum.arg(2, (m * n) as i32).unwrap();
            sum.arg(3, nsplit as i32).unwrap();
            g.launch(&sum, [(m * n).div_ceil(256) as u32, 1, 1])
                .unwrap();
        }
        let mut out = vec![0u8; m * n * 4];
        g.download(&dc, &mut out).unwrap();
        let c: Vec<f32> = out
            .as_chunks::<4>()
            .0
            .iter()
            .map(|x| f32::from_le_bytes(*x))
            .collect();
        let (mut worst, mut big) = (0.0f64, 0.0f64);
        for r in 0..m {
            for col in 0..n {
                let s: f64 = (0..k)
                    .map(|i| a[r * k + i] as f64 * wf[col * k + i] as f64)
                    .sum();
                worst = worst.max((c[r * n + col] as f64 - s).abs());
                big = big.max(s.abs());
            }
        }
        println!("{m8} m={m} split={nsplit}: worst abs err {worst:.3e} (max |C| {big:.2})");
        assert!(
            worst < 2e-3 * big,
            "{m8} m={m} split={nsplit} mismatch {worst}"
        );
    }
    // timing at the 27B verify shapes
    for (n, k) in [
        (17408usize, 5120usize),
        (5120, 17408),
        (10240, 5120),
        (5120, 6144),
    ] {
        let m = 8usize;
        let wq = random_quant(fmt, n, k, &mut seed);
        let a: Vec<f32> = (0..m * k).map(|_| lcg(&mut seed)).collect();
        let (da, dw, dp, dc) = (
            g.alloc(m * k * 2).unwrap(),
            g.alloc(wq.len()).unwrap(),
            g.alloc(16 * m * n * 4).unwrap(),
            g.alloc(m * n * 4).unwrap(),
        );
        g.upload(&da, &f16_bytes(&a)).unwrap();
        g.upload(&dw, &wq).unwrap();
        let mut line = format!("{m8} m=8 {k}->{n} ({} MiB):", wq.len() >> 20);
        for nsplit in [1usize, 2, 4, 8] {
            if k / 256 < nsplit * 2 {
                continue;
            }
            k8.arg_buf(0, &da).unwrap();
            k8.arg_buf(1, &dw).unwrap();
            k8.arg_buf(2, if nsplit > 1 { &dp } else { &dc }).unwrap();
            k8.arg(3, m as i32).unwrap();
            k8.arg(4, n as i32).unwrap();
            k8.arg(5, k as i32).unwrap();
            k8.arg(6, nsplit as i32).unwrap();
            let mut t = g
                .time_launches(&k8, [n.div_ceil(256) as u32, nsplit as u32, 1], 50)
                .unwrap();
            t.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let mut med = t[t.len() / 2];
            if nsplit > 1 {
                sum.arg_buf(0, &dp).unwrap();
                sum.arg_buf(1, &dc).unwrap();
                sum.arg(2, (m * n) as i32).unwrap();
                sum.arg(3, nsplit as i32).unwrap();
                let mut ts = g
                    .time_launches(&sum, [(m * n).div_ceil(256) as u32, 1, 1], 50)
                    .unwrap();
                ts.sort_by(|x, y| x.partial_cmp(y).unwrap());
                med += ts[ts.len() / 2];
            }
            line += &format!(
                "  s{nsplit} {:6.1} us {:4.0} GB/s",
                med / 1e3,
                wq.len() as f64 / med
            );
        }
        println!("{line}");
    }
}

#[test]
#[ignore = "requires a Level Zero GPU"]
fn skinny_q4k_probe() {
    let fmt = QFormat {
        name: "gemm_q4k_m8",
        dtype: infr_core::DType::Q4K,
        block_bytes: 144,
        block_vals: 256,
        f16_fields: &[0, 2],
        wg_m: 8,
        wg_n: 256,
    };
    skinny_probe(
        &fmt,
        infr_ze::kernels::gemm_kquant(),
        "gemm_q4k_m8",
        "gemm_q4k_m16",
    );
}

#[test]
#[ignore = "requires a Level Zero GPU"]
fn skinny_q5k_probe() {
    let fmt = QFormat {
        name: "gemm_q5k_m8",
        dtype: infr_core::DType::Q5K,
        block_bytes: 176,
        block_vals: 256,
        f16_fields: &[0, 2],
        wg_m: 8,
        wg_n: 256,
    };
    skinny_probe(
        &fmt,
        infr_ze::kernels::gemm_kquant(),
        "gemm_q5k_m8",
        "gemm_q5k_m16",
    );
}

#[test]
#[ignore = "requires a Level Zero GPU"]
fn skinny_q6k_probe() {
    let fmt = QFormat {
        name: "gemm_q6k_m8",
        dtype: infr_core::DType::Q6K,
        block_bytes: 210,
        block_vals: 256,
        f16_fields: &[208],
        wg_m: 8,
        wg_n: 256,
    };
    skinny_probe(
        &fmt,
        infr_ze::kernels::gemm_kq16(),
        "gemm_q6k_m8",
        "gemm_q6k_m16",
    );
}

#[test]
#[ignore = "requires a Level Zero GPU"]
fn skinny_q3k_probe() {
    let fmt = QFormat {
        name: "gemm_q3k_m8",
        dtype: infr_core::DType::Q3K,
        block_bytes: 110,
        block_vals: 256,
        f16_fields: &[108],
        wg_m: 8,
        wg_n: 256,
    };
    skinny_probe(
        &fmt,
        infr_ze::kernels::gemm_kq16(),
        "gemm_q3k_m8",
        "gemm_q3k_m16",
    );
}

#[test]
#[ignore = "requires a Level Zero GPU"]
fn skinny_iq4xs_probe() {
    let fmt = QFormat {
        name: "gemm_iq4xs_m8",
        dtype: infr_core::DType::Iq4Xs,
        block_bytes: 136,
        block_vals: 256,
        f16_fields: &[0],
        wg_m: 8,
        wg_n: 256,
    };
    skinny_probe(
        &fmt,
        infr_ze::kernels::gemm_iq4(),
        "gemm_iq4xs_m8",
        "gemm_iq4xs_m16",
    );
}

#[test]
#[ignore = "requires a Level Zero GPU"]
fn skinny_iq4nl_probe() {
    let fmt = QFormat {
        name: "gemm_iq4nl_m8",
        dtype: infr_core::DType::Iq4Nl,
        block_bytes: 18,
        block_vals: 32,
        f16_fields: &[0],
        wg_m: 8,
        wg_n: 256,
    };
    skinny_probe(
        &fmt,
        infr_ze::kernels::gemm_iq4(),
        "gemm_iq4nl_m8",
        "gemm_iq4nl_m16",
    );
}

/// Host cost of appending a tiny kernel to the immediate command list (what a Level Zero chain pays
/// per op), then of the device catching up.
#[test]
#[ignore = "requires a Level Zero GPU"]
fn launch_overhead_probe() {
    let (Some(g), Some(ew)) = (gpu(), infr_ze::kernels::elementwise()) else {
        eprintln!("skip: no Level Zero GPU or no ocloc kernels");
        return;
    };
    let m = g.module(ew).unwrap();
    let k = m.kernel("sum_splits").unwrap();
    k.group_size(256, 1, 1).unwrap();
    let (p, c) = (g.alloc(4096 * 4).unwrap(), g.alloc(4096 * 4).unwrap());
    k.arg_buf(0, &p).unwrap();
    k.arg_buf(1, &c).unwrap();
    k.arg(2, 4096i32).unwrap();
    k.arg(3, 1i32).unwrap();
    for n in [10usize, 200, 1000] {
        let t0 = std::time::Instant::now();
        for _ in 0..n {
            g.launch(&k, [16, 1, 1]).unwrap();
        }
        let appended = t0.elapsed();
        let mut out = vec![0u8; 16];
        g.download(&c, &mut out).unwrap(); // synchronizes
        println!(
            "{n} launches: append {:.1} us each, append+drain {:.1} us each",
            appended.as_secs_f64() * 1e6 / n as f64,
            t0.elapsed().as_secs_f64() * 1e6 / n as f64
        );
    }
}
