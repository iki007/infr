//! DFlash2 block-diffusion speculative decoding (`docs/dflash.md`).
//!
//! A DFlash2 drafter (`general.architecture = dflash`, e.g. z-lab's Qwen3.8-27B-DFlash2) is a small
//! non-causal transformer that drafts a whole block of tokens in ONE forward. It never sees token
//! history directly: after every target forward, the inputs of a few target layers
//! (`dflash.target_layers`, the residual stream entering them) are fused by `fc` + `enc.output_norm`
//! and written into each drafter layer's K/V cache at the target positions ("injection"). A draft
//! then runs the block `[anchor, mask × (block-1)]` through the drafter, attending bidirectionally
//! to the block and to the injected context, and a low-rank pairwise selector turns the block's
//! top-k candidates into a lattice that a greedy walk reads the draft off.
//!
//! The math follows llama.cpp's `src/models/dflash.cpp` (`graph<false>`, `build_dflash2_conv`,
//! `build_dflash2_selector`) and `common/speculative.cpp` (`common_speculative_impl_draft_dflash`).
//! The target side is [`crate::seam::SeamKv::enable_taps`]; the speculative loop is
//! [`generate_dflash_spec_core`].

use anyhow::{anyhow, bail, ensure, Result};
use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage};
use infr_core::graph::{Activation, AttnMask, Graph, Op};
use infr_core::loader::MetaValue;
use infr_core::tensor::{DType, TensorDesc, TensorId};
use infr_core::WeightSource;
use infr_gguf::Gguf;
use std::collections::hash_map::Entry;
use std::path::Path;

/// A sampled draft: the tokens and, per step, the proposal distribution `q` they were drawn from.
pub type SampledDraft = (Vec<u32>, Vec<Vec<(u32, f32)>>);

/// The drafter's hyperparameters, read from the `dflash.*` GGUF keys.
#[derive(Clone, Debug)]
pub struct DflashHp {
    pub n_layer: usize,
    pub ne: usize,
    pub nff: usize,
    pub nh: usize,
    pub nkv: usize,
    pub hd: usize,
    pub eps: f32,
    pub theta: f32,
    /// Tokens per draft block (anchor + `block - 1` mask tokens).
    pub block: usize,
    pub conv_kernel: usize,
    pub conv_group: usize,
    pub sel_rank: usize,
    pub sel_top_k: usize,
    /// Target layers whose INPUT feeds the drafter (llama.cpp's `dflash.target_layers`).
    pub target_layers: Vec<usize>,
    /// Sliding window over the injected context; 0 = full attention.
    pub window: usize,
    pub mask_token: u32,
    /// Target vocabulary (the selector tables and `token_embd` rows).
    pub vocab: usize,
    /// Drafter head width (the reduced vocabulary `d2t` maps into the target's).
    pub draft_vocab: usize,
}

/// A loaded, shape-checked DFlash2 drafter GGUF. The weights stay in the mmap until a
/// [`DflashSession`] uploads them; the embedding and selector tables are read row by row on the
/// host.
pub struct DflashDrafter {
    g: Gguf,
    pub hp: DflashHp,
    /// Reduced-vocab index → target token id.
    d2t: Vec<u32>,
}

const N_GLOBAL: usize = 5;
const N_PER_LAYER: usize = 15;

fn find<'a>(g: &'a Gguf, name: &str) -> Result<&'a infr_core::TensorInfo> {
    g.tensors()
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| anyhow!("DFlash GGUF: missing tensor {name}"))
}

fn require(g: &Gguf, name: &str, want: &[usize]) -> Result<()> {
    let t = find(g, name)?;
    ensure!(
        t.shape == want,
        "DFlash GGUF: {name} has shape {:?}, expected {:?}",
        t.shape,
        want
    );
    Ok(())
}

impl DflashDrafter {
    /// Open a DFlash2 drafter and check it against the `target` model it will draft for.
    pub fn open(path: &Path, target: &crate::Config) -> Result<Self> {
        let g = Gguf::open(path).map_err(|e| anyhow!("{}: {e}", path.display()))?;
        let md = g.metadata();
        let arch = md.str("general.architecture").unwrap_or("");
        ensure!(
            arch == "dflash",
            "{}: architecture `{arch}`, expected `dflash`",
            path.display()
        );
        let u = |k: &str| -> Result<usize> {
            md.u64(&format!("dflash.{k}"))
                .map(|v| v as usize)
                .ok_or_else(|| anyhow!("DFlash GGUF: missing dflash.{k}"))
        };
        let f = |k: &str| -> Result<f32> {
            md.get(&format!("dflash.{k}"))
                .and_then(MetaValue::as_f64)
                .map(|v| v as f32)
                .ok_or_else(|| anyhow!("DFlash GGUF: missing dflash.{k}"))
        };
        let arr = |k: &str| -> Option<Vec<MetaValue>> {
            md.get(&format!("dflash.{k}"))
                .and_then(MetaValue::as_arr)
                .map(<[MetaValue]>::to_vec)
        };
        let target_layers: Vec<usize> = arr("target_layers")
            .ok_or_else(|| anyhow!("DFlash GGUF: missing dflash.target_layers"))?
            .iter()
            .map(|v| v.as_u64().map(|x| x as usize))
            .collect::<Option<_>>()
            .ok_or_else(|| anyhow!("DFlash GGUF: dflash.target_layers is not integers"))?;
        ensure!(
            !target_layers.is_empty(),
            "DFlash GGUF: empty target_layers"
        );
        if let Some(&l) = target_layers.iter().find(|&&l| l >= target.n_layer) {
            bail!(
                "DFlash drafter taps target layer {l}, but the target has {} layers — wrong \
                 drafter for this model",
                target.n_layer
            );
        }
        let n_layer = u("block_count")?;
        // Every layer sliding-window (the shipped drafters) or none of them; a mixed pattern
        // would need per-layer windows.
        let window = match arr("attention.sliding_window_pattern") {
            Some(p) if p.iter().all(|v| *v == MetaValue::Bool(true)) => {
                u("attention.sliding_window")?
            }
            Some(p) if p.iter().all(|v| *v == MetaValue::Bool(false)) => 0,
            Some(_) => bail!("DFlash GGUF: mixed sliding-window pattern is not supported"),
            None => md
                .u64("dflash.attention.sliding_window")
                .map_or(0, |v| v as usize),
        };
        // M-RoPE sections on a drafter for an M-RoPE target: text positions feed the same value to
        // every section, which is plain NeoX — unless a section reads the 4th (always 0) slot.
        if let Some(s) = arr("rope.dimension_sections") {
            ensure!(
                s.get(3).and_then(MetaValue::as_u64).unwrap_or(0) == 0,
                "DFlash GGUF: an M-RoPE section on the 4th position axis is not supported"
            );
        }
        let ne = u("embedding_length")?;
        let hd = u("attention.key_length")?;
        ensure!(
            u("attention.value_length")? == hd,
            "DFlash GGUF: key_length != value_length"
        );
        let hp = DflashHp {
            n_layer,
            ne,
            nff: u("feed_forward_length")?,
            nh: u("attention.head_count")?,
            nkv: u("attention.head_count_kv")?,
            hd,
            eps: f("attention.layer_norm_rms_epsilon")?,
            theta: f("rope.freq_base")?,
            block: u("block_size")?,
            conv_kernel: u("conv_kernel_size")?,
            conv_group: u("conv_group_size")?,
            sel_rank: u("selector_rank")?,
            sel_top_k: u("selector_top_k")?,
            target_layers,
            window,
            mask_token: md
                .u64("tokenizer.ggml.mask_token_id")
                .ok_or_else(|| anyhow!("DFlash GGUF: missing tokenizer.ggml.mask_token_id"))?
                as u32,
            vocab: target.vocab,
            draft_vocab: find(&g, "output.weight")?
                .shape
                .get(1)
                .copied()
                .unwrap_or(0),
        };
        ensure!(hp.block >= 2, "DFlash GGUF: block_size {} < 2", hp.block);
        ensure!(
            hp.ne.is_multiple_of(hp.conv_group) && hp.nh.is_multiple_of(hp.nkv),
            "DFlash GGUF: inconsistent head/group sizes"
        );
        let n_tap = hp.target_layers.len();
        ensure!(
            hp.ne == target.n_embd,
            "DFlash drafter width {} != target width {} — wrong drafter for this model",
            hp.ne,
            target.n_embd
        );
        let (q, kv, proj) = (
            hp.nh * hd,
            hp.nkv * hd,
            2 * hp.conv_kernel * (ne / hp.conv_group),
        );
        require(&g, "token_embd.weight", &[ne, hp.vocab])?;
        require(&g, "fc.weight", &[n_tap * target.n_embd, ne])?;
        require(&g, "enc.output_norm.weight", &[ne])?;
        require(&g, "output_norm.weight", &[ne])?;
        require(&g, "output.weight", &[ne, hp.draft_vocab])?;
        require(&g, "selector_hidden.weight", &[ne, hp.sel_rank])?;
        require(&g, "selector_predecessor.weight", &[hp.sel_rank, hp.vocab])?;
        require(&g, "selector_successor.weight", &[hp.sel_rank, hp.vocab])?;
        require(&g, "d2t", &[hp.draft_vocab])?;
        for l in 0..n_layer {
            let p = |s: &str| format!("blk.{l}.{s}");
            require(&g, &p("attn_norm.weight"), &[ne])?;
            require(&g, &p("attn_conv_proj.weight"), &[ne, proj])?;
            require(&g, &p("attn_conv_base"), &[ne, hp.conv_kernel, 2])?;
            require(&g, &p("attn_q.weight"), &[ne, q])?;
            require(&g, &p("attn_k.weight"), &[ne, kv])?;
            require(&g, &p("attn_v.weight"), &[ne, kv])?;
            require(&g, &p("attn_q_norm.weight"), &[hd])?;
            require(&g, &p("attn_k_norm.weight"), &[hd])?;
            require(&g, &p("attn_output.weight"), &[q, ne])?;
            require(&g, &p("ffn_norm.weight"), &[ne])?;
            require(&g, &p("ffn_conv_proj.weight"), &[ne, proj])?;
            require(&g, &p("ffn_conv_base"), &[ne, hp.conv_kernel, 2])?;
            require(&g, &p("ffn_gate.weight"), &[ne, hp.nff])?;
            require(&g, &p("ffn_up.weight"), &[ne, hp.nff])?;
            require(&g, &p("ffn_down.weight"), &[hp.nff, ne])?;
            for b in ["attn_conv_base", "ffn_conv_base"] {
                ensure!(
                    find(&g, &p(b))?.dtype == DType::F32,
                    "DFlash GGUF: {} must be f32",
                    p(b)
                );
            }
        }
        for t in ["selector_predecessor.weight", "selector_successor.weight"] {
            ensure!(
                find(&g, t)?.dtype == DType::Q8_0 || find(&g, t)?.dtype == DType::F32,
                "DFlash GGUF: {t} must be q8_0 or f32"
            );
        }
        let d2t_info = find(&g, "d2t")?;
        let bytes = g.tensor_bytes("d2t").map_err(|e| anyhow!("{e}"))?;
        let d2t: Vec<u32> = match d2t_info.dtype {
            DType::I64 => bytes
                .chunks_exact(8)
                .map(|b| i64::from_le_bytes(b.try_into().expect("8 bytes")))
                .map(|v| u32::try_from(v).ok().filter(|&t| (t as usize) < hp.vocab))
                .collect::<Option<_>>(),
            DType::I32 => bytes
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes(b.try_into().expect("4 bytes")))
                .map(|v| u32::try_from(v).ok().filter(|&t| (t as usize) < hp.vocab))
                .collect::<Option<_>>(),
            dt => bail!("DFlash GGUF: d2t is {dt:?}, expected i64"),
        }
        .ok_or_else(|| anyhow!("DFlash GGUF: d2t holds an id outside the target vocabulary"))?;
        ensure!(d2t.len() == hp.draft_vocab, "DFlash GGUF: d2t length");
        ensure!(
            (hp.mask_token as usize) < hp.vocab,
            "DFlash GGUF: mask token outside the vocabulary"
        );
        Ok(Self { g, hp, d2t })
    }

    /// The weight upload order [`DflashSession`] uses (and its graphs declare): the five global
    /// tensors, then [`N_PER_LAYER`] per layer.
    fn weight_names(&self) -> Vec<String> {
        let mut v: Vec<String> = [
            "fc.weight",
            "enc.output_norm.weight",
            "output_norm.weight",
            "output.weight",
            "selector_hidden.weight",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        for l in 0..self.hp.n_layer {
            for s in [
                "attn_norm.weight",
                "attn_conv_proj.weight",
                "attn_conv_base",
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_q_norm.weight",
                "attn_k_norm.weight",
                "attn_output.weight",
                "ffn_norm.weight",
                "ffn_conv_proj.weight",
                "ffn_conv_base",
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
            ] {
                v.push(format!("blk.{l}.{s}"));
            }
        }
        debug_assert_eq!(v.len(), N_GLOBAL + N_PER_LAYER * self.hp.n_layer);
        v
    }

    /// Row `row` of a `[width, rows]` table, dequantized on the host.
    fn table_row(&self, name: &str, row: usize, width: usize) -> Result<Vec<f32>> {
        let t = find(&self.g, name)?;
        let rb = infr_gguf::nbytes(t.dtype, width);
        let bytes = self.g.tensor_bytes(name).map_err(|e| anyhow!("{e}"))?;
        let src = bytes
            .get(row * rb..(row + 1) * rb)
            .ok_or_else(|| anyhow!("DFlash: {name} row {row} out of range"))?;
        crate::quant::dequant_block(t.dtype, src).map_err(|e| anyhow!("{name}: {e}"))
    }

    /// Bytes the drafter's device weights take.
    pub fn weight_bytes(&self) -> usize {
        self.weight_names()
            .iter()
            .filter_map(|n| find(&self.g, n).ok())
            .map(|t| t.nbytes)
            .sum()
    }
}

/// Uploads one GGUF tensor to the backend (the per-backend binder the MTP head uses too).
pub(crate) type BindWeightFn<'a> =
    dyn Fn(&str, crate::seam::WBytes, DType, usize) -> Result<(Box<dyn Buffer>, DType)> + 'a;

/// A drafter resident on a backend: its weights, its per-layer K/V caches (a ring over the
/// sliding window when the backend supports one), and the timing counters the spec loop reports.
pub struct DflashSession {
    d: std::sync::Arc<DflashDrafter>,
    wbufs: Vec<Box<dyn Buffer>>,
    wspecs: Vec<(DType, usize)>,
    kc: Vec<Box<dyn Buffer>>,
    vc: Vec<Box<dyn Buffer>>,
    kv_rows: usize,
    max_ctx: usize,
    mask_row: Vec<f32>,
    pub inject_secs: f64,
    pub draft_secs: f64,
}

struct Common {
    positions: TensorId,
    kc: Vec<TensorId>,
    vc: Vec<TensorId>,
    w: Vec<TensorId>,
}

impl DflashSession {
    /// Upload `d`'s weights through `bind` (onto `be`) and allocate K/V for a session of up to
    /// `max_ctx` positions. Every later call must pass the same backend.
    pub(crate) fn new(
        be: &dyn Backend,
        d: std::sync::Arc<DflashDrafter>,
        bind: &BindWeightFn,
        max_ctx: usize,
    ) -> Result<Self> {
        let mut wbufs = Vec::new();
        let mut wspecs = Vec::new();
        for name in d.weight_names() {
            let t = find(&d.g, &name)?;
            let tb = d.g.tensor_bytes_arc(&name).map_err(|e| anyhow!("{e}"))?;
            let numel: usize = t.shape.iter().product();
            let (buf, dt) = bind(&name, crate::seam::WBytes::Mmap(tb), t.dtype, numel)?;
            wbufs.push(buf);
            wspecs.push((dt, numel));
        }
        let hp = &d.hp;
        // The window plus one block of keys is all a draft reads, so a ring of that size (and a
        // little slack for an injection batch) never recycles a visible row.
        let kv_rows = if hp.window > 0 && be.capabilities().kv_swa_ring {
            (hp.window + hp.block + 64).min(max_ctx + hp.block)
        } else {
            max_ctx + hp.block
        };
        let kvrow = hp.nkv * hp.hd;
        let mut kc = Vec::new();
        let mut vc = Vec::new();
        for _ in 0..hp.n_layer {
            kc.push(
                be.alloc(kv_rows * kvrow * 2, BufferUsage::KvCache)
                    .map_err(|e| anyhow!("{e}"))?,
            );
            vc.push(
                be.alloc(kv_rows * kvrow * 2, BufferUsage::KvCache)
                    .map_err(|e| anyhow!("{e}"))?,
            );
        }
        let mask_row = d.table_row("token_embd.weight", hp.mask_token as usize, hp.ne)?;
        Ok(Self {
            d,
            wbufs,
            wspecs,
            kc,
            vc,
            kv_rows,
            max_ctx,
            mask_row,
            inject_secs: 0.0,
            draft_secs: 0.0,
        })
    }

    pub fn hp(&self) -> &DflashHp {
        &self.d.hp
    }

    fn declare(&self, g: &mut Graph, rows: usize) -> Common {
        let hp = &self.d.hp;
        let kvrow = hp.nkv * hp.hd;
        let positions = g.input(TensorDesc::new(vec![rows], DType::I32));
        let kc = (0..hp.n_layer)
            .map(|_| g.input(TensorDesc::new(vec![self.kv_rows * kvrow], DType::F16)))
            .collect();
        let vc = (0..hp.n_layer)
            .map(|_| g.input(TensorDesc::new(vec![self.kv_rows * kvrow], DType::F16)))
            .collect();
        let w = self
            .wspecs
            .iter()
            .map(|&(dt, n)| g.weight(TensorDesc::new(vec![n], dt)))
            .collect();
        Common {
            positions,
            kc,
            vc,
            w,
        }
    }

    fn bind_common<'b>(&'b self, b: &mut Bindings<'b>, c: &Common, pos: &'b dyn Buffer) {
        b.bind(c.positions, pos);
        for l in 0..self.d.hp.n_layer {
            b.bind(c.kc[l], self.kc[l].as_ref());
            b.bind(c.vc[l], self.vc[l].as_ref());
        }
        for (i, &w) in c.w.iter().enumerate() {
            b.bind(w, self.wbufs[i].as_ref());
        }
    }

    /// The positions the session was sized for.
    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    fn positions_buf(
        &self,
        be: &dyn Backend,
        start: usize,
        rows: usize,
    ) -> Result<Box<dyn Buffer>> {
        let p: Vec<i32> = (start..start + rows).map(|x| x as i32).collect();
        let buf = be
            .alloc(rows * 4, BufferUsage::Staging)
            .map_err(|e| anyhow!("{e}"))?;
        be.upload(buf.as_ref(), bytemuck::cast_slice(&p))
            .map_err(|e| anyhow!("{e}"))?;
        Ok(buf)
    }

    /// Write the drafter K/V for target positions `start..start+n` from the target's tap ring
    /// (`ring` holds `ring_rows` slots, slot `pos % ring_rows` = position `pos`).
    pub fn inject(
        &mut self,
        be: &dyn Backend,
        ring: &dyn Buffer,
        ring_rows: usize,
        start: usize,
        n: usize,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        ensure!(
            n <= ring_rows,
            "DFlash inject: {n} rows exceed the {ring_rows}-slot tap ring"
        );
        ensure!(
            start + n <= self.max_ctx + self.d.hp.block,
            "DFlash inject: position {} past the session's {} positions",
            start + n,
            self.max_ctx
        );
        let t0 = std::time::Instant::now();
        const CHUNK: usize = 512;
        let mut s = start;
        while s < start + n {
            let rows = (start + n - s).min(CHUNK);
            self.inject_chunk(be, ring, ring_rows, s, rows)?;
            s += rows;
        }
        self.inject_secs += t0.elapsed().as_secs_f64();
        Ok(())
    }

    fn inject_chunk(
        &mut self,
        be: &dyn Backend,
        ring: &dyn Buffer,
        ring_rows: usize,
        start: usize,
        rows: usize,
    ) -> Result<()> {
        let hp = &self.d.hp;
        let (ne, kvrow) = (hp.ne, hp.nkv * hp.hd);
        let wf = hp.target_layers.len() * ne;
        let mut g = Graph::new();
        let f32d = |n: usize| TensorDesc::new(vec![n], DType::F32);
        let c = self.declare(&mut g, rows);
        let ring_id = g.input(f32d(ring_rows * wf));
        let feat = g.internal(f32d(rows * wf));
        let g0 = g.internal(f32d(rows * ne));
        let gn = g.internal(f32d(rows * ne));
        let k = g.internal(f32d(rows * kvrow));
        let k16 = g.internal(TensorDesc::new(vec![rows * kvrow], DType::F16));
        let v = g.internal(f32d(rows * kvrow));
        let mut r = 0;
        while r < rows {
            let slot = (start + r) % ring_rows;
            let n = (rows - r).min(ring_rows - slot);
            g.push(Op::CopyStrided {
                src: ring_id,
                src_off: (slot * wf) as u32,
                src_stride: wf as u32,
                dst: feat,
                dst_off: (r * wf) as u32,
                dst_stride: wf as u32,
                rows: n as u32,
                n: wf as u32,
            });
            r += n;
        }
        g.push(Op::Linear {
            x: feat,
            weight: c.w[0],
            dst: g0,
            m: rows as u32,
            in_f: wf as u32,
            out_f: ne as u32,
            w_off: 0,
        });
        g.push(Op::RmsNorm {
            x: g0,
            weight: c.w[1],
            dst: gn,
            rows: rows as u32,
            dim: ne as u32,
            eps: hp.eps,
        });
        for l in 0..hp.n_layer {
            let w = |i: usize| c.w[N_GLOBAL + N_PER_LAYER * l + i];
            g.push(Op::Linear {
                x: gn,
                weight: w(4),
                dst: k,
                m: rows as u32,
                in_f: ne as u32,
                out_f: kvrow as u32,
                w_off: 0,
            });
            g.push(Op::QkNormRope {
                x: k,
                weight: w(7),
                positions: c.positions,
                dst: k16,
                rows: rows as u32,
                n_head: hp.nkv as u32,
                head_dim: hp.hd as u32,
                rope_dim: hp.hd as u32,
                theta: hp.theta,
                eps: hp.eps,
                freq_factors: None,
                x_stride: 0,
            });
            g.push(Op::WriteKv {
                src: k16,
                cache: c.kc[l],
                rows: rows as u32,
                row_stride: kvrow as u32,
                pos: start as u32,
            });
            g.push(Op::Linear {
                x: gn,
                weight: w(5),
                dst: v,
                m: rows as u32,
                in_f: ne as u32,
                out_f: kvrow as u32,
                w_off: 0,
            });
            g.push(Op::WriteKv {
                src: v,
                cache: c.vc[l],
                rows: rows as u32,
                row_stride: kvrow as u32,
                pos: start as u32,
            });
        }
        let plan = be.compile(&g).map_err(|e| anyhow!("{e}"))?;
        let pos = self.positions_buf(be, start, rows)?;
        let mut b = Bindings::new();
        self.bind_common(&mut b, &c, pos.as_ref());
        b.bind(ring_id, ring);
        be.execute(plan.as_ref(), &b).map_err(|e| anyhow!("{e}"))?;
        Ok(())
    }

    /// One block forward: `[anchor, mask × (block-1)]` at positions `pos0..pos0+block`, over the
    /// injected context `[lo, pos0)`. Returns the head logits `[block, draft_vocab]` and the
    /// selector gate `[block, rank]`.
    fn block_forward(
        &mut self,
        be: &dyn Backend,
        anchor: u32,
        pos0: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let hp = self.d.hp.clone();
        let (ne, b_) = (hp.ne, hp.block);
        let (q, kvrow) = (hp.nh * hp.hd, hp.nkv * hp.hd);
        let proj = 2 * hp.conv_kernel * (ne / hp.conv_group);
        ensure!(
            pos0 + b_ <= self.max_ctx + b_,
            "DFlash draft: position {pos0} past the session's {} positions",
            self.max_ctx
        );
        let lo = if hp.window > 0 {
            (pos0 + 1).saturating_sub(hp.window)
        } else {
            0
        };
        let mut g = Graph::new();
        let f32d = |n: usize| TensorDesc::new(vec![n], DType::F32);
        let c = self.declare(&mut g, b_);
        let inp = g.input(f32d(b_ * ne));
        let logits = g.output(f32d(b_ * hp.draft_vocab));
        let gate = g.output(f32d(b_ * hp.sel_rank));
        let x = g.internal(f32d(b_ * ne)); // residual stream
        let nn = g.internal(f32d(b_ * ne));
        let nc = g.internal(f32d(b_ * ne));
        let dynb = g.internal(f32d(b_ * proj));
        let qb = g.internal(f32d(b_ * q));
        let kb = g.internal(f32d(b_ * kvrow));
        let vb = g.internal(f32d(b_ * kvrow));
        let q16 = g.internal(TensorDesc::new(vec![b_ * q], DType::F16));
        let k16 = g.internal(TensorDesc::new(vec![b_ * kvrow], DType::F16));
        let attn = g.internal(f32d(b_ * q));
        let o = g.internal(f32d(b_ * ne));
        let ffn_inp = g.internal(f32d(b_ * ne));
        let gb = g.internal(f32d(b_ * hp.nff));
        let ub = g.internal(f32d(b_ * hp.nff));
        let ab = g.internal(f32d(b_ * hp.nff));
        let rn = g.internal(f32d(b_ * ne));
        let m = b_ as u32;
        let lin = |g: &mut Graph, x: TensorId, w: TensorId, dst: TensorId, i: usize, o: usize| {
            g.push(Op::Linear {
                x,
                weight: w,
                dst,
                m,
                in_f: i as u32,
                out_f: o as u32,
                w_off: 0,
            });
        };
        let conv = |g: &mut Graph,
                    x: TensorId,
                    coeff: TensorId,
                    base: TensorId,
                    dst: TensorId,
                    side: u32| {
            g.push(Op::BlockConv {
                x,
                coeff,
                base,
                dst,
                rows: m,
                dim: ne as u32,
                group: hp.conv_group as u32,
                kernel: hp.conv_kernel as u32,
                block: m,
                side,
            });
        };
        let norm = |g: &mut Graph, x: TensorId, w: TensorId, dst: TensorId| {
            g.push(Op::RmsNorm {
                x,
                weight: w,
                dst,
                rows: m,
                dim: ne as u32,
                eps: hp.eps,
            });
        };
        g.push(Op::Copy {
            src: inp,
            src_off: 0,
            dst: x,
            dst_off: 0,
            n: (b_ * ne) as u32,
        });
        for l in 0..hp.n_layer {
            let w = |i: usize| c.w[N_GLOBAL + N_PER_LAYER * l + i];
            norm(&mut g, x, w(0), nn);
            lin(&mut g, nn, w(1), dynb, ne, proj);
            conv(&mut g, nn, dynb, w(2), nc, 0);
            lin(&mut g, nc, w(3), qb, ne, q);
            lin(&mut g, nc, w(4), kb, ne, kvrow);
            lin(&mut g, nc, w(5), vb, ne, kvrow);
            g.push(Op::QkNormRope {
                x: kb,
                weight: w(7),
                positions: c.positions,
                dst: k16,
                rows: m,
                n_head: hp.nkv as u32,
                head_dim: hp.hd as u32,
                rope_dim: hp.hd as u32,
                theta: hp.theta,
                eps: hp.eps,
                freq_factors: None,
                x_stride: 0,
            });
            g.push(Op::WriteKv {
                src: k16,
                cache: c.kc[l],
                rows: m,
                row_stride: kvrow as u32,
                pos: pos0 as u32,
            });
            g.push(Op::WriteKv {
                src: vb,
                cache: c.vc[l],
                rows: m,
                row_stride: kvrow as u32,
                pos: pos0 as u32,
            });
            g.push(Op::QkNormRope {
                x: qb,
                weight: w(6),
                positions: c.positions,
                dst: q16,
                rows: m,
                n_head: hp.nh as u32,
                head_dim: hp.hd as u32,
                rope_dim: hp.hd as u32,
                theta: hp.theta,
                eps: hp.eps,
                freq_factors: None,
                x_stride: 0,
            });
            g.push(Op::Attention {
                q: q16,
                k_cache: c.kc[l],
                v_cache: c.vc[l],
                dst: attn,
                rows: m,
                kv_len: (pos0 + b_) as u32,
                n_head: hp.nh as u32,
                n_kv: hp.nkv as u32,
                head_dim: hp.hd as u32,
                scale: 1.0 / (hp.hd as f32).sqrt(),
                mask: AttnMask::Canvas { lo },
                pos: pos0 as u32,
                sinks: None,
                key_bias: None,
            });
            lin(&mut g, attn, w(8), o, q, ne);
            conv(&mut g, o, dynb, w(2), nc, 1);
            g.push(Op::Add {
                a: nc,
                b: x,
                dst: ffn_inp,
                n: (b_ * ne) as u32,
            });
            norm(&mut g, ffn_inp, w(9), nn);
            lin(&mut g, nn, w(10), dynb, ne, proj);
            conv(&mut g, nn, dynb, w(11), nc, 0);
            lin(&mut g, nc, w(12), gb, ne, hp.nff);
            lin(&mut g, nc, w(13), ub, ne, hp.nff);
            g.push(Op::GatedAct {
                gate: gb,
                up: ub,
                dst: ab,
                rows: m,
                nff: hp.nff as u32,
                act: Activation::Silu,
                up_off: 0,
                up_stride: 0,
                gate_stride: 0,
                gate_block_width: 0,
                swiglu_clamp: None,
            });
            lin(&mut g, ab, w(14), o, hp.nff, ne);
            conv(&mut g, o, dynb, w(11), nc, 1);
            g.push(Op::Add {
                a: nc,
                b: ffn_inp,
                dst: x,
                n: (b_ * ne) as u32,
            });
        }
        norm(&mut g, x, c.w[2], rn);
        lin(&mut g, rn, c.w[3], logits, ne, hp.draft_vocab);
        lin(&mut g, rn, c.w[4], gate, ne, hp.sel_rank);

        let plan = be.compile(&g).map_err(|e| anyhow!("{e}"))?;
        let mut rows = self.d.table_row("token_embd.weight", anchor as usize, ne)?;
        for _ in 1..b_ {
            rows.extend_from_slice(&self.mask_row);
        }
        let inp_buf = be
            .alloc(b_ * ne * 4, BufferUsage::Staging)
            .map_err(|e| anyhow!("{e}"))?;
        be.upload(inp_buf.as_ref(), bytemuck::cast_slice(&rows))
            .map_err(|e| anyhow!("{e}"))?;
        let pos = self.positions_buf(be, pos0, b_)?;
        // Device-local outputs, read back with one copy: the head's scattered stores into
        // host-visible memory are slow on some GPUs (see `SeamKv::logits_dev`).
        let lbuf = be
            .alloc(b_ * hp.draft_vocab * 4, BufferUsage::Activations)
            .map_err(|e| anyhow!("{e}"))?;
        let gbuf = be
            .alloc(b_ * hp.sel_rank * 4, BufferUsage::Activations)
            .map_err(|e| anyhow!("{e}"))?;
        let mut b = Bindings::new();
        self.bind_common(&mut b, &c, pos.as_ref());
        b.bind(inp, inp_buf.as_ref());
        b.bind(logits, lbuf.as_ref());
        b.bind(gate, gbuf.as_ref());
        be.execute(plan.as_ref(), &b).map_err(|e| anyhow!("{e}"))?;
        let mut lv = vec![0f32; b_ * hp.draft_vocab];
        be.download(lbuf.as_ref(), bytemuck::cast_slice_mut(&mut lv))
            .map_err(|e| anyhow!("{e}"))?;
        let mut gv = vec![0f32; b_ * hp.sel_rank];
        be.download(gbuf.as_ref(), bytemuck::cast_slice_mut(&mut gv))
            .map_err(|e| anyhow!("{e}"))?;
        Ok((lv, gv))
    }

    /// Draft up to `n` (≤ block-1) tokens after `anchor`, the last committed token, which sits at
    /// position `pos0`. The drafter context must already hold every position below `pos0`.
    /// Greedy: each step takes the lattice's best successor of the previous pick.
    pub fn draft(
        &mut self,
        be: &dyn Backend,
        anchor: u32,
        pos0: usize,
        n: usize,
    ) -> Result<Vec<u32>> {
        Ok(self.draft_impl(be, anchor, pos0, n, None)?.0)
    }

    /// [`draft`](Self::draft) that SAMPLES each step from `softmax(scores / temp)` over the
    /// step's candidates instead, returning each step's proposal distribution `q` alongside —
    /// what the speculative-sampling accept rule needs.
    pub fn draft_sample(
        &mut self,
        be: &dyn Backend,
        anchor: u32,
        pos0: usize,
        n: usize,
        temp: f32,
        rng: &mut u64,
    ) -> Result<SampledDraft> {
        self.draft_impl(be, anchor, pos0, n, Some((temp, rng)))
    }

    fn draft_impl(
        &mut self,
        be: &dyn Backend,
        anchor: u32,
        pos0: usize,
        n: usize,
        mut sample: Option<(f32, &mut u64)>,
    ) -> Result<SampledDraft> {
        let t0 = std::time::Instant::now();
        let hp = self.d.hp.clone();
        let n = n.min(hp.block - 1);
        if n == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let (logits, gate) = self.block_forward(be, anchor, pos0)?;
        let (dv, rank, k) = (hp.draft_vocab, hp.sel_rank, hp.sel_top_k);
        // Per block position 1..=n: the top-k candidates (target ids) and their logits.
        let mut cands: Vec<Vec<u32>> = Vec::with_capacity(n + 1);
        let mut unary: Vec<Vec<f32>> = Vec::with_capacity(n + 1);
        cands.push(vec![anchor]);
        unary.push(vec![0.0]);
        for i in 1..=n {
            let row = &logits[i * dv..(i + 1) * dv];
            let top = top_k(row, k);
            unary.push(top.iter().map(|&j| row[j]).collect());
            cands.push(top.iter().map(|&j| self.d.d2t[j]).collect());
        }
        // Selector rows, cached per token for this draft.
        let mut succ = std::collections::HashMap::<u32, Vec<f32>>::new();
        let mut pred = std::collections::HashMap::<u32, Vec<f32>>::new();
        let mut out = Vec::with_capacity(n);
        let mut qs = Vec::new();
        let mut p = 0usize; // index into the previous position's candidates
        for i in 1..=n {
            let prev = cands[i - 1][p];
            if let Entry::Vacant(e) = pred.entry(prev) {
                e.insert(
                    self.d
                        .table_row("selector_predecessor.weight", prev as usize, rank)?,
                );
            }
            let gi = &gate[i * rank..(i + 1) * rank];
            let cond: Vec<f32> = pred[&prev].iter().zip(gi).map(|(a, b)| a * b).collect();
            let mut scores = Vec::with_capacity(cands[i].len());
            for (j, &tok) in cands[i].iter().enumerate() {
                if let Entry::Vacant(e) = succ.entry(tok) {
                    e.insert(
                        self.d
                            .table_row("selector_successor.weight", tok as usize, rank)?,
                    );
                }
                let s: f32 = succ[&tok]
                    .iter()
                    .zip(&cond)
                    .map(|(a, b)| a * b)
                    .sum::<f32>();
                scores.push(s + unary[i][j]);
            }
            // First maximum, like `std::max_element`.
            let mut best = 0usize;
            for (j, &s) in scores.iter().enumerate() {
                if s > scores[best] {
                    best = j;
                }
            }
            p = best;
            if let Some((temp, rng)) = sample.as_mut() {
                let t = temp.max(1e-6);
                let mut probs: Vec<f32> = scores
                    .iter()
                    .map(|&s| ((s - scores[best]) / t).exp())
                    .collect();
                let sum: f32 = probs.iter().sum();
                probs.iter_mut().for_each(|x| *x /= sum);
                let mut u = crate::sampling::next_uniform(rng);
                for (j, &pj) in probs.iter().enumerate() {
                    u -= pj;
                    if u < 0.0 {
                        p = j;
                        break;
                    }
                }
                qs.push(cands[i].iter().copied().zip(probs).collect());
            }
            out.push(cands[i][p]);
        }
        self.draft_secs += t0.elapsed().as_secs_f64();
        Ok((out, qs))
    }
}

/// Indices of the `k` largest entries of `row`, largest first (ties: lower index first).
fn top_k(row: &[f32], k: usize) -> Vec<usize> {
    let mut best: Vec<(f32, usize)> = Vec::with_capacity(k + 1);
    for (i, &v) in row.iter().enumerate() {
        if best.len() == k && v <= best[k - 1].0 {
            continue;
        }
        let at = best.partition_point(|&(b, _)| b >= v);
        best.insert(at, (v, i));
        best.truncate(k);
    }
    best.into_iter().map(|(_, i)| i).collect()
}

/// Per-run counters of [`generate_dflash_spec_core`] (the prompt prefill excluded).
#[derive(Debug, Clone, Copy, Default)]
pub struct DflashTiming {
    pub draft_secs: f64,
    pub verify_secs: f64,
    pub inject_secs: f64,
    /// DeltaNet snapshots/restores.
    pub other_secs: f64,
    /// Boundary re-primes (re-prefilling the committed suffix after a rollback).
    pub reprime_secs: f64,
    pub reprimes: usize,
    pub cycles: usize,
    pub total_drafted: usize,
    pub total_accepted: usize,
}

impl DflashTiming {
    /// Accepted drafts / drafted.
    pub fn alpha(&self) -> f64 {
        self.total_accepted as f64 / self.total_drafted.max(1) as f64
    }
    /// Tokens committed per target verify (accepted drafts + the target's own token).
    pub fn tokens_per_cycle(&self) -> f64 {
        (self.total_accepted + self.cycles) as f64 / self.cycles.max(1) as f64
    }
}

fn argmax_row(row: &[f32]) -> u32 {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > bv {
            bv = v;
            bi = i;
        }
    }
    bi as u32
}

/// One target VERIFY over `feed`'s uncached suffix: per-row argmax ids (greedy) or the full
/// `[m, vocab]` logits (`full`), plus `m`, the rows it processed (positions `feed.len()-m ..`).
#[allow(clippy::too_many_arguments)]
fn target_verify(
    be: &dyn Backend,
    bind: &BindWeightFn,
    model: &crate::SeamModel,
    feed: &[u32],
    trunk: &mut Option<crate::seam::SeamKv>,
    max_ctx: usize,
    full: bool,
) -> Result<(Vec<u32>, Vec<f32>, usize)> {
    let cfg = model.config();
    let mut logits = Vec::new();
    let mut ids = Vec::new();
    let (_, stats) = crate::seam::generate_dense_backend(
        be,
        bind,
        model.gguf(),
        cfg,
        model.engine_cfg(),
        model.embd(),
        None,
        feed,
        0,
        |_| {},
        trunk,
        max_ctx,
        None,
        Some(&mut logits),
        if full { None } else { Some(&mut ids) },
        None,
        None,
        None,
        None,
    )?;
    let m = stats.n_prompt;
    if !full && ids.is_empty() {
        ensure!(logits.len() == m * cfg.vocab, "DFlash verify: logits shape");
        ids = (0..m)
            .map(|j| argmax_row(&logits[j * cfg.vocab..(j + 1) * cfg.vocab]))
            .collect();
    }
    Ok((ids, logits, m))
}

/// Prefill `tokens`' uncached suffix through the ordinary chunked path and sample the next token.
/// Returns `(token, logits of the last row when asked, rows processed)`.
fn target_prime(
    be: &dyn Backend,
    bind: &BindWeightFn,
    model: &crate::SeamModel,
    tokens: &[u32],
    trunk: &mut Option<crate::seam::SeamKv>,
    max_ctx: usize,
    want_logits: bool,
) -> Result<(u32, Vec<f32>, usize)> {
    let mut logits = Vec::new();
    let (ids, stats) = crate::seam::generate_dense_backend(
        be,
        bind,
        model.gguf(),
        model.config(),
        model.engine_cfg(),
        model.embd(),
        None,
        tokens,
        1,
        |_| {},
        trunk,
        max_ctx,
        None,
        None,
        None,
        want_logits.then_some(&mut logits),
        None,
        None,
        None,
    )?;
    ensure!(ids.len() == 1, "DFlash prime: expected one sampled token");
    Ok((ids[0], logits, stats.n_prompt))
}

/// Inject target positions `lo..hi` from the trunk's tap ring into the drafter.
fn inject_range(
    be: &dyn Backend,
    sess: &mut DflashSession,
    trunk: &Option<crate::seam::SeamKv>,
    lo: usize,
    hi: usize,
) -> Result<()> {
    if hi <= lo {
        return Ok(());
    }
    let t = trunk
        .as_ref()
        .and_then(|st| st.taps())
        .ok_or_else(|| anyhow!("DFlash: the target session has no taps"))?;
    sess.inject(be, t.buf.as_ref(), t.rows, lo, hi - lo)
}

/// The leading prediction a re-prime hands the next verify (the row the verify then skips).
enum Lead {
    Id(u32),
    Dist(Vec<(u32, f32)>),
}

/// The DFlash draft → verify → inject loop over a caller-held target session `trunk` and drafter
/// session `sess` (both on `be`). Greedy (`sampling.temp <= 0`) commits exactly the target's
/// argmax stream; `temp > 0` uses the speculative-sampling accept rule with the lattice's
/// per-step proposal distributions. The DeltaNet rollback (`spec.mtp_ckpt`) and boundary re-prime
/// (`spec.mtp_reprime`) are MTP's, unchanged (see `mtp::generate_mtp_spec_core`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_dflash_spec_core(
    be: &dyn Backend,
    bind: &BindWeightFn,
    model: &crate::SeamModel,
    sess: &mut DflashSession,
    trunk: &mut Option<crate::seam::SeamKv>,
    max_ctx: usize,
    prompt: &str,
    max_new: usize,
    mut on_piece: impl FnMut(&str),
) -> Result<(crate::GenStats, DflashTiming)> {
    let cfg = model.config();
    let ec = model.engine_cfg();
    let hp = sess.hp().clone();
    ensure!(
        hp.window > 0,
        "DFlash: drafters without a sliding window are not supported yet"
    );
    let ring_rows = hp.window + hp.block + 64;
    let n_max = hp.block - 1;
    // DeltaNet rollback after a partial accept: the verify records each row's recurrent state
    // (`SeamKv::spec_hist`) and the accepted row's state is selected directly, so every verify is
    // exactly `1 + drafts` rows. A target without DeltaNet layers needs no rollback at all; MTP's
    // snapshot + restore + re-prime (`spec.mtp_ckpt`) stays as the fallback should a history
    // ever be unavailable.
    let mut mtp_ckpt = ec.spec.mtp_ckpt;
    const REPRIME_MAX_M: usize = 16;
    let ignore_eos = ec.sampling.ignore_eos;
    let hit_eos = |t: u32| !ignore_eos && (cfg.eos_ids.contains(&t) || t == cfg.eos);
    let sampler = crate::sampling::Sampler::from_cfg(&ec.sampling);
    let mut rng = crate::sampling::seed_rng(&ec.sampling);
    let stochastic = sampler.temp > 0.0;
    let prof = ec.prof.stages;

    let toks = model.encode(prompt)?;
    ensure!(!toks.is_empty(), "DFlash: empty prompt");
    let p = toks.len();

    // The target session must exist before its taps can be switched on: a cold start primes one
    // token first, then the reset below re-prefills the whole prompt with the taps live.
    if trunk.is_none() {
        target_prime(be, bind, model, &toks[..1], trunk, max_ctx, false)?;
    }
    {
        let st = trunk.as_mut().expect("trunk primed above");
        st.enable_taps(be, cfg, &hp.target_layers, ring_rows)?;
        st.reset();
    }
    let hist = trunk
        .as_mut()
        .expect("trunk primed above")
        .enable_spec_hist(be, cfg, hp.block)?;
    if hist {
        mtp_ckpt = false;
    }
    let reprime = mtp_ckpt && ec.spec.mtp_reprime;

    // ── prime: prefill the prompt, sample the first token, seed the drafter context ──────────
    let t_prime = std::time::Instant::now();
    let (t0, _, m0) = target_prime(be, bind, model, &toks, trunk, max_ctx, false)?;
    // The first draft (anchor at position p) reads positions from p+1-window on.
    inject_range(be, sess, trunk, p.saturating_sub(hp.window).max(p - m0), p)?;
    if mtp_ckpt {
        if let Some(st) = trunk.as_mut() {
            st.mtp_snapshot_delta(be, cfg)?;
        }
    }
    let mut boundary = p;
    let prompt_secs = t_prime.elapsed().as_secs_f64();

    let mut acc: Vec<u32> = Vec::new();
    let mut printed = 0usize;
    let mut out: Vec<u32> = Vec::new();
    let mut committed = toks.clone();
    let mut tm = DflashTiming::default();
    let t_decode = std::time::Instant::now();
    let inject0 = sess.inject_secs;
    let draft0 = sess.draft_secs;

    let mut push = |c: u32, out: &mut Vec<u32>, committed: &mut Vec<u32>| -> bool {
        let eos = hit_eos(c);
        out.push(c);
        committed.push(c);
        if !eos {
            crate::stream_token(model.tokenizer(), &mut acc, &mut printed, c, &mut on_piece);
        }
        eos || out.len() >= max_new
    };
    let mut lead: Option<Lead> = None;
    if !push(t0, &mut out, &mut committed) {
        'cycles: loop {
            let n_round = n_max.min(max_new - out.len());
            if n_round == 0 {
                break;
            }
            tm.cycles += 1;
            let pos0 = committed.len() - 1;
            let anchor = committed[pos0];
            let (cand, q_dists) = if stochastic {
                sess.draft_sample(be, anchor, pos0, n_round, sampler.temp, &mut rng)?
            } else {
                (sess.draft(be, anchor, pos0, n_round)?, Vec::new())
            };

            let mut feed = committed.clone();
            feed.extend_from_slice(&cand);
            let t_verify = std::time::Instant::now();
            let (vids, vlogits, m) =
                target_verify(be, bind, model, &feed, trunk, max_ctx, stochastic)?;
            let verify_secs = t_verify.elapsed().as_secs_f64();
            tm.verify_secs += verify_secs;
            let first = feed.len() - m;
            let lead_now = lead.take();
            let base = match lead_now {
                Some(_) => {
                    ensure!(m == cand.len(), "DFlash verify: leading row shape");
                    0
                }
                None => {
                    ensure!(
                        m > cand.len(),
                        "DFlash verify: got {m} rows for {} drafts",
                        cand.len()
                    );
                    m - (cand.len() + 1)
                }
            };
            let (accepted, next_tok) = if stochastic {
                let vocab = cfg.vocab;
                let mut p_rows: Vec<Vec<(u32, f32)>> = Vec::with_capacity(cand.len() + 1);
                if let Some(Lead::Dist(d)) = lead_now {
                    p_rows.push(d);
                }
                p_rows.extend((base..m).map(|j| {
                    crate::sampling::truncated_dist(&vlogits[j * vocab..(j + 1) * vocab], sampler)
                }));
                crate::seam::model::spec_accept_stochastic(&cand, &q_dists, &p_rows, &mut rng)
            } else {
                let mut varg: Vec<u32> = Vec::with_capacity(cand.len() + 1);
                if let Some(Lead::Id(id)) = lead_now {
                    varg.push(id);
                }
                varg.extend_from_slice(&vids[base..]);
                crate::seam::model::spec_accept(&cand, &varg)
            };
            tm.total_drafted += cand.len();
            tm.total_accepted += accepted;

            // Seed the drafter with every verified row whose token stays committed.
            inject_range(be, sess, trunk, first, pos0 + accepted + 1)?;

            let t_other = std::time::Instant::now();
            let mut restored = false;
            if hist {
                ensure!(
                    base == 0 && m == cand.len() + 1,
                    "DFlash: a {m}-row verify for {} drafts with the state history on",
                    cand.len()
                );
                trunk
                    .as_mut()
                    .expect("trunk exists")
                    .spec_hist_select(be, accepted + 1)?;
            } else if mtp_ckpt {
                if let Some(st) = trunk.as_mut() {
                    if accepted == cand.len() {
                        st.mtp_snapshot_delta(be, cfg)?;
                        boundary = feed.len();
                    } else {
                        st.mtp_restore_delta(be)?;
                        restored = true;
                    }
                }
            }
            tm.other_secs += t_other.elapsed().as_secs_f64();
            if prof {
                tracing::info!(
                    "[dflash cycle {}] drafted={} accepted={accepted} m={m} verify={:.1}ms",
                    tm.cycles,
                    cand.len(),
                    verify_secs * 1e3
                );
            }

            for &c in &cand[..accepted] {
                if push(c, &mut out, &mut committed) {
                    break 'cycles;
                }
            }
            if push(next_tok, &mut out, &mut committed) {
                break 'cycles;
            }

            if restored && reprime && (committed.len() - boundary) + n_max > REPRIME_MAX_M {
                let t_re = std::time::Instant::now();
                let (rid, rlogits, rm) =
                    target_prime(be, bind, model, &committed, trunk, max_ctx, stochastic)?;
                inject_range(be, sess, trunk, committed.len() - rm, committed.len() - 1)?;
                lead = Some(if stochastic {
                    ensure!(rlogits.len() == cfg.vocab, "DFlash reprime: logits shape");
                    Lead::Dist(crate::sampling::truncated_dist(&rlogits, sampler))
                } else {
                    Lead::Id(rid)
                });
                if let Some(st) = trunk.as_mut() {
                    st.mtp_snapshot_delta(be, cfg)?;
                }
                boundary = committed.len();
                tm.reprime_secs += t_re.elapsed().as_secs_f64();
                tm.reprimes += 1;
            }
        }
    }
    tm.inject_secs = sess.inject_secs - inject0;
    tm.draft_secs = sess.draft_secs - draft0;
    if prof {
        tracing::info!(
            "[dflash summary] {} cycles, {}/{} accepted (alpha={:.3}, {:.2} tokens/cycle), {} tokens; \
             draft {:.0}ms verify {:.0}ms inject {:.0}ms other {:.0}ms",
            tm.cycles,
            tm.total_accepted,
            tm.total_drafted,
            tm.alpha(),
            tm.tokens_per_cycle(),
            out.len(),
            tm.draft_secs * 1e3,
            tm.verify_secs * 1e3,
            tm.inject_secs * 1e3,
            tm.other_secs * 1e3,
        );
    }
    Ok((
        crate::GenStats {
            n_prompt: p,
            prompt_secs,
            n_gen: out.len(),
            decode_secs: t_decode.elapsed().as_secs_f64(),
        },
        tm,
    ))
}

/// The smallest target context a DFlash session allocates (matches MTP's trunk minimum).
const TRUNK_MIN_CTX: usize = 16384;

/// DFlash speculative decode of `prompt` on the Vulkan backend `vk`, reusing `trunk` (the target
/// session) and `sess` (the drafter session) when they are large enough — a chat keeps both across
/// turns so neither model's weights are uploaded twice.
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_dflash_spec_vulkan_on(
    vk: &infr_vulkan::VulkanBackend,
    model: &crate::SeamModel,
    drafter: &std::sync::Arc<DflashDrafter>,
    sess: &mut Option<DflashSession>,
    trunk: &mut Option<crate::seam::SeamKv>,
    prompt: &str,
    max_new: usize,
    on_piece: impl FnMut(&str),
) -> Result<(crate::GenStats, DflashTiming)> {
    let need = model.encode(prompt)?.len() + max_new + drafter.hp.block + 8;
    if trunk.as_ref().is_some_and(|t| t.max_ctx() < need) {
        *trunk = None;
    }
    let max_ctx = match trunk.as_ref() {
        Some(t) => t.max_ctx(),
        None => need.max(TRUNK_MIN_CTX),
    };
    let bind: &BindWeightFn = &|_name, tb, dt, _n| {
        let bytes = tb.materialize();
        let padded = infr_vulkan::linear::pad_to_u32_align(&bytes);
        let buf = vk
            .alloc(padded.len(), BufferUsage::Weights)
            .map_err(|e| anyhow!("{e}"))?;
        vk.upload(buf.as_ref(), &padded)
            .map_err(|e| anyhow!("{e}"))?;
        Ok((buf, dt))
    };
    if sess.as_ref().is_some_and(|s| s.max_ctx() < max_ctx) {
        *sess = None;
    }
    if sess.is_none() {
        *sess = Some(DflashSession::new(
            vk,
            std::sync::Arc::clone(drafter),
            bind,
            max_ctx,
        )?);
    }
    generate_dflash_spec_core(
        vk,
        bind,
        model,
        sess.as_mut().expect("just ensured"),
        trunk,
        max_ctx,
        prompt,
        max_new,
        on_piece,
    )
}

/// [`generate_dflash_spec_vulkan_on`] on a fresh Vulkan backend and target session.
pub fn generate_dflash_spec_vulkan(
    model: &crate::SeamModel,
    drafter: &std::sync::Arc<DflashDrafter>,
    prompt: &str,
    max_new: usize,
    on_piece: impl FnMut(&str),
) -> Result<(crate::GenStats, DflashTiming)> {
    let vk = infr_vulkan::VulkanBackend::new_with(model.cfg().clone())
        .map_err(|e| anyhow!("vulkan init: {e}"))?;
    generate_dflash_spec_vulkan_on(
        &vk, model, drafter, &mut None, &mut None, prompt, max_new, on_piece,
    )
}

#[cfg(test)]
mod tests {
    use super::top_k;

    #[test]
    fn top_k_orders_and_breaks_ties_low_index_first() {
        let row = [0.5, 3.0, -1.0, 3.0, 2.0, f32::NEG_INFINITY];
        assert_eq!(top_k(&row, 3), vec![1, 3, 4]);
        assert_eq!(top_k(&row, 1), vec![1]);
        assert_eq!(top_k(&row, 10), vec![1, 3, 4, 0, 2, 5]);
    }
}
