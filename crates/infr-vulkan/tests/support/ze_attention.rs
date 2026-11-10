//! Unit-test support for attention selection and both cross-API handoffs, including the arena's
//! f16 query and shadowed bound f32 output. Run serially on the headless card with
//! `INFR_ZE_VK=1 cargo test --release -p infr-vulkan ze_attention -- --ignored --test-threads=1`.
use super::*;
use infr_testkit::{f16_bytes, gen_f32};
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
struct Case {
    rows: usize,
    kv_len: usize,
    capacity: usize,
    nh: usize,
    nkv: usize,
    pos: usize,
    window: usize,
    scale: f32,
}

fn reference(c: Case, q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; q.len()];
    let scale = if c.scale > 0.0 {
        c.scale as f64
    } else {
        1.0 / 16.0
    };
    for r in 0..c.rows {
        let end = (c.pos + r + 1).min(c.kv_len);
        let begin = if c.window > 0 {
            (c.pos + r + 1).saturating_sub(c.window)
        } else {
            0
        };
        for h in 0..c.nh {
            let kh = h / (c.nh / c.nkv);
            let qi = (r * c.nh + h) * 256;
            let scores: Vec<f64> = (begin..end)
                .map(|j| {
                    let ki = ((j % c.capacity) * c.nkv + kh) * 256;
                    (0..256)
                        .map(|d| q[qi + d] as f64 * k[ki + d] as f64)
                        .sum::<f64>()
                        * scale
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
            let sum: f64 = weights.iter().sum();
            if sum == 0.0 {
                continue;
            }
            for d in 0..256 {
                out[qi + d] = (weights
                    .iter()
                    .zip(begin..end)
                    .map(|(w, j)| w * v[((j % c.capacity) * c.nkv + kh) * 256 + d] as f64)
                    .sum::<f64>()
                    / sum) as f32;
            }
        }
    }
    out
}

#[test]
#[ignore = "needs an Intel Xe2 GPU with both Mesa ANV and Level Zero"]
fn ze_attention_matches_vulkan_and_host() {
    let idx = std::env::var("INFR_ZE_VK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let backend = |enabled| {
        let mut cfg = infr_core::config::Config::default();
        cfg.kernels.vulkan.ze_gemm = enabled;
        VulkanBackend::new_on_with(idx, Arc::new(cfg)).unwrap()
    };
    let vk = backend(false);
    let ze = backend(true);
    assert!(
        ze.ze_bridge().is_some(),
        "Level Zero bridge must be active for this test"
    );
    for c in [
        Case {
            rows: 64,
            kv_len: 64,
            capacity: 64,
            nh: 2,
            nkv: 1,
            pos: 0,
            window: 0,
            scale: 0.0,
        },
        Case {
            rows: 100,
            kv_len: 300,
            capacity: 320,
            nh: 6,
            nkv: 2,
            pos: 200,
            window: 0,
            scale: 0.0,
        },
        Case {
            rows: 130,
            kv_len: 260,
            capacity: 288,
            nh: 4,
            nkv: 4,
            pos: 130,
            window: 70,
            scale: 0.2,
        },
        // All keys masked for the first row; later rows still have no visible keys.
        Case {
            rows: 65,
            kv_len: 40,
            capacity: 64,
            nh: 2,
            nkv: 1,
            pos: 200,
            window: 16,
            scale: 0.0,
        },
        // A wrapped sliding-window cache must stay on Vulkan, even with the bridge enabled.
        Case {
            rows: 65,
            kv_len: 400,
            capacity: 128,
            nh: 2,
            nkv: 1,
            pos: 335,
            window: 60,
            scale: 0.0,
        },
    ] {
        let rounded = |n, seed| {
            gen_f32(n, seed)
                .into_iter()
                .map(|x| half::f16::from_f32(x).to_f32())
                .collect::<Vec<_>>()
        };
        let q = rounded(c.rows * c.nh * 256, 7);
        let k = rounded(c.capacity * c.nkv * 256, 11);
        let v = rounded(c.capacity * c.nkv * 256, 19);
        let want = reference(c, &q, &k, &v);
        let mut graph = Graph::new();
        let qin = graph.input(TensorDesc::new(vec![c.rows, c.nh, 256], DType::F16));
        let qi = graph.internal(graph.desc(qin).clone());
        let ki = graph.input(TensorDesc::new(vec![c.capacity, c.nkv, 256], DType::F16));
        let vi = graph.input(graph.desc(ki).clone());
        let oi = graph.output(TensorDesc::new(vec![c.rows, c.nh, 256], DType::F32));
        graph.push(Op::Copy {
            src: qin,
            src_off: 0,
            dst: qi,
            dst_off: 0,
            n: q.len() as u32,
        });
        graph.push(Op::Attention {
            q: qi,
            k_cache: ki,
            v_cache: vi,
            dst: oi,
            rows: c.rows as u32,
            kv_len: c.kv_len as u32,
            n_head: c.nh as u32,
            n_kv: c.nkv as u32,
            head_dim: 256,
            scale: c.scale,
            pos: c.pos as u32,
            mask: if c.window > 0 {
                AttnMask::SlidingWindow(c.window)
            } else {
                AttnMask::Causal
            },
            sinks: None,
            key_bias: None,
        });
        for be_ in [&vk, &ze] {
            let qb = be_.alloc(q.len() * 2, BufferUsage::Activations).unwrap();
            let kb = be_.alloc(k.len() * 2, BufferUsage::KvCache).unwrap();
            let vb = be_.alloc(v.len() * 2, BufferUsage::KvCache).unwrap();
            let ob = be_.alloc(want.len() * 4, BufferUsage::Activations).unwrap();
            be_.upload(qb.as_ref(), &f16_bytes(&q)).unwrap();
            be_.upload(kb.as_ref(), &f16_bytes(&k)).unwrap();
            be_.upload(vb.as_ref(), &f16_bytes(&v)).unwrap();
            let mut bindings = Bindings::new();
            bindings.bind(qin, qb.as_ref());
            bindings.bind(ki, kb.as_ref());
            bindings.bind(vi, vb.as_ref());
            bindings.bind(oi, ob.as_ref());
            let expected = std::ptr::eq(be_, &ze) && c.capacity >= c.kv_len;
            assert_eq!(
                ze_plan(be_, &graph, &bindings).contains(&1),
                expected,
                "{c:?}"
            );
            // Unsupported attention semantics must not enter the Level Zero chain.
            if expected {
                for variant in 0..3 {
                    let mut unsupported = graph.clone();
                    if let Op::Attention {
                        mask,
                        sinks,
                        key_bias,
                        ..
                    } = &mut unsupported.ops[1]
                    {
                        match variant {
                            0 => *mask = AttnMask::Canvas { lo: 0 },
                            1 => *sinks = Some(oi),
                            _ => *key_bias = Some(oi),
                        }
                    }
                    assert!(ze_plan(be_, &unsupported, &bindings).is_empty());
                }
            }
            let plan = be_.compile(&graph).unwrap();
            // A second forward also exercises arena reuse and advancing semaphore values.
            for _ in 0..2 {
                be_.execute(plan.as_ref(), &bindings).unwrap();
                let mut got = vec![0.0f32; want.len()];
                be_.download(ob.as_ref(), bytemuck::cast_slice_mut(&mut got))
                    .unwrap();
                for (i, (&actual, &expected)) in got.iter().zip(&want).enumerate() {
                    assert!(
                        actual.is_finite() && (actual - expected).abs() < 2e-3,
                        "{c:?}, ze={}, element {i}: got {actual}, want {expected}",
                        std::ptr::eq(be_, &ze)
                    );
                }
            }
        }
        println!("attention parity: {c:?}");
    }
}
