//! Level Zero prefill kernels between Vulkan segments — the hybrid of backlog B102
//! (`docs/level-zero.md`). Default on for Intel Xe2; `INFR_ZE_GEMM=0` disables it.
//!
//! On an Intel Xe2 card driven by both Mesa ANV and the Intel compute runtime, the prefill GEMMs
//! of the formats `infr-ze` has DPAS kernels for, their joining elementwise ops, and compatible
//! head-dim-256 prefill attention run on the same card. The remaining ops stay on Vulkan.
//!
//! - **Memory**: the weight blocks (`bda_weight_alloc`) and an *exchange arena* holding the
//!   offloaded kernels' activations, plus the KV caches, are dedicated exportable allocations
//!   (`alloc_exportable`); Level Zero imports each once as a dma-buf (keyed by its `export_id`) and
//!   addresses tensors as base + offset. Graph tensors in the arena are ordinary `BdaSub` views,
//!   so every Vulkan op reads and writes them unchanged.
//! - **Ordering**: one Vulkan timeline semaphore, imported into Level Zero. At an offloaded GEMM the
//!   adapter submits the segment so far signalling `v`; Level Zero waits `v`, runs the GEMM and
//!   signals `v + 1`; the next segment's submit waits `v + 1`. Nothing waits on the host.
//!
//! Linux only (the interop rides dma-buf fds); elsewhere [`ZeCell`] never enables.

use infr_core::backend::Buffer;
use infr_core::error::Result;
use infr_core::DType;

use crate::VulkanBackend;

/// Can `dt` run as an offloaded GEMM (a Level Zero kernel exists for it)?
pub(crate) fn ze_dtype(dt: DType) -> bool {
    weight_block(dt).is_some()
}

/// `(values, bytes)` of `dt`'s block as the Level Zero kernels walk it, and the K multiple they
/// need: Q6_K / Q3_K / IQ4_NL blocks are 2-byte multiples, walked in dword-aligned PAIRS.
fn weight_block(dt: DType) -> Option<(usize, usize, usize)> {
    Some(match dt {
        DType::Q4K => (256, 144, 256),
        DType::Q5K => (256, 176, 256),
        DType::Iq4Xs => (256, 136, 256),
        DType::Q6K => (256, 210, 512),
        DType::Q3K => (256, 110, 512),
        DType::Iq4Nl => (32, 18, 64),
        _ => return None,
    })
}

/// The shapes the kernels take: K a multiple of the format's block (pair), and either a
/// prefill-sized M (the tiled GEMMs: N a multiple of their 64-column tile) or a verify-sized one —
/// 2..=16 rows, a speculative verify or a DFlash draft block (the skinny GEMMs, opt-in with
/// `skinny`: N a multiple of 32). Single-row decode stays on Vulkan.
pub(crate) fn ze_shape(dt: DType, m: usize, in_f: usize, out_f: usize, skinny: bool) -> bool {
    let prefill = weight_block(dt).is_some_and(|(_, _, kmul)| {
        m >= 256 && out_f.is_multiple_of(64) && in_f.is_multiple_of(kmul)
    });
    let small = skinny
        && (2..=SKINNY_MAX_M).contains(&m)
        && out_f.is_multiple_of(4)
        && skinny_block(dt).is_some_and(|(_, _, kmul)| in_f.is_multiple_of(kmul));
    prefill || small
}

/// [`weight_block`] for the skinny GEMMs, which also cover Q8_0 (34-byte blocks of 32, walked in
/// dword-aligned pairs).
fn skinny_block(dt: DType) -> Option<(usize, usize, usize)> {
    match dt {
        DType::Q8_0 => Some((32, 34, 64)),
        DType::Q4_0 => Some((32, 18, 64)),
        _ => weight_block(dt),
    }
}

/// Can an op of `rows` rows run as one of the verify-sized Level Zero kernels (the chain that
/// carries a whole speculative-verify layer, `INFR_ZE_SKINNY`)?
pub(crate) fn ze_small(rows: usize, skinny: bool) -> bool {
    skinny && (2..=SKINNY_MAX_M).contains(&rows)
}

/// The profile label of a skinny GEMM.
fn skinny_label(dt: DType) -> &'static str {
    match dt {
        DType::Q4K => "gemm_sk_q4k",
        DType::Q5K => "gemm_sk_q5k",
        DType::Q6K => "gemm_sk_q6k",
        DType::Q3K => "gemm_sk_q3k",
        DType::Iq4Xs => "gemm_sk_iq4xs",
        DType::Iq4Nl => "gemm_sk_iq4nl",
        DType::Q8_0 => "gemm_sk_q8_0",
        DType::Q4_0 => "gemm_sk_q4_0",
        _ => "gemm_sk",
    }
}

/// One argument of a [`ZeBridge::run`] launch.
pub(crate) enum ZArg<'a> {
    B(&'a dyn Buffer),
    /// A buffer at a further byte offset.
    O(&'a dyn Buffer, usize),
    I(i32),
    F(f32),
}

/// The skinny GEMMs' row cap (two DPAS row blocks).
const SKINNY_MAX_M: usize = 16;

/// K splits for a skinny GEMM: wide outputs already give every subgroup its own weight rows, a
/// narrow one (a 5120-wide projection is 160 subgroups) splits K four ways so enough of them stream
/// at once. Measured on the B70 at m=8: Q4_K 17408->5120 166 us unsplit, 97 us in 4 (517 GB/s).
fn skinny_splits(dt: DType, in_f: usize, out_f: usize) -> usize {
    let (vals, _, kmul) = skinny_block(dt).expect("ze_shape admitted the dtype");
    // Split units: superblocks, superblock pairs, or IQ4_NL / Q8_0 block pairs.
    let units = in_f / kmul.max(vals);
    if out_f >= 32768 || units < 8 {
        1
    } else if out_f >= 1024 {
        4
    } else {
        // A narrow output (the 48-wide DeltaNet beta/alpha projections: two subgroups' worth of
        // columns) streams almost nothing per subgroup, so it is latency-bound: split K as far
        // as two units a split, up to 16 ways (measured 26 us a call unsplit-ish at 4).
        (units / 2).clamp(4, 16)
    }
}

/// Bytes of the skinny GEMMs' split-K partials buffer: 4 splits of [16, 32768) f32 (the up-to-16
/// splits of a narrow output fit in the same bytes).
const SKINNY_PARTIALS: usize = 4 * SKINNY_MAX_M * 32768 * 4;

#[cfg(target_os = "linux")]
mod imp {
    use super::*;
    use crate::{as_vk_buf, be, Backing, BdaBlockHandle, TpExportSemaphore, VkBuffer};
    use ash::vk;
    use gpu_allocator::MemoryLocation;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};

    /// The bridge and the exchange arena, owned by the [`VulkanBackend`] handle (never
    /// `VulkanShared`: both hold `Arc<VulkanShared>` clones — the cycle `moe_pager`'s doc warns of).
    #[derive(Default)]
    pub(crate) struct ZeCell {
        bridge: OnceLock<Option<ZeBridge>>,
        arena: Mutex<Option<Arena>>,
    }

    /// The exchange arena: one exportable block, bump-allocated per forward.
    struct Arena {
        handle: Arc<BdaBlockHandle>,
        size: usize,
        cursor: usize,
        export_id: u64,
    }

    /// Rows per Level Zero launch — the kernels' workgroup tile height (see `ZeBridge::gemm`).
    const PACKET: usize = 256;

    pub(crate) struct ZeBridge(Mutex<Inner>);

    // SAFETY: every Level Zero object — and the `Rc`s `infr-ze` keeps them in — is created, used and
    // dropped only under the mutex (or by the bridge's own drop), and none escapes it.
    unsafe impl Send for ZeBridge {}
    unsafe impl Sync for ZeBridge {}

    struct Inner {
        gpu: infr_ze::Gpu,
        q4k: infr_ze::Kernel,
        q5k: infr_ze::Kernel,
        iq4xs: infr_ze::Kernel,
        q6k: infr_ze::Kernel,
        q3k: infr_ze::Kernel,
        iq4nl: infr_ze::Kernel,
        attn256: infr_ze::Kernel,
        rmsnorm: infr_ze::Kernel,
        swiglu: infr_ze::Kernel,
        add: infr_ze::Kernel,
        to_f16: infr_ze::Kernel,
        /// The skinny (2..=16-row) GEMMs by `(dtype, two row blocks?)`.
        skinny: HashMap<(DType, bool), infr_ze::Kernel>,
        sum_splits: infr_ze::Kernel,
        /// Split-K partials of the skinny GEMMs ([split][M][N] f32), Level Zero-owned.
        partials: infr_ze::DevBuf,
        /// The verify-sized layer ops (`decode.cl`, `attn_dec.cl`) by kernel name.
        dec: HashMap<&'static str, infr_ze::Kernel>,
        /// `attn_dec256`'s chunk partials (o, then (m, l)), grown on demand.
        attn_parts: Option<(infr_ze::DevBuf, infr_ze::DevBuf, usize)>,
        /// The Vulkan side of the timeline (signalled by segment submits, waited by the next).
        vk_sem: TpExportSemaphore,
        /// The same timeline imported into Level Zero.
        ze_sem: infr_ze::Semaphore,
        /// The last value handed out (`reserve`).
        value: u64,
        /// Level Zero imports of exportable allocations, by `export_id`.
        imports: HashMap<u64, infr_ze::DevBuf>,
    }

    impl ZeBridge {
        fn open(be_: &VulkanBackend) -> std::result::Result<ZeBridge, String> {
            let pci = be_
                .shared
                .pci_bdf
                .clone()
                .ok_or("no VK_EXT_pci_bus_info to find the Level Zero device")?;
            let ze = infr_ze::Ze::open().map_err(|e| e.to_string())?;
            let gpu = infr_ze::Gpu::open(&ze, Some(&pci)).map_err(|e| e.to_string())?;
            let kq = infr_ze::kernels::gemm_kquant().ok_or("infr-ze built without ocloc")?;
            let iq = infr_ze::kernels::gemm_iq4().ok_or("infr-ze built without ocloc")?;
            let ew = infr_ze::kernels::elementwise().ok_or("infr-ze built without ocloc")?;
            let k16 = infr_ze::kernels::gemm_kq16().ok_or("infr-ze built without ocloc")?;
            let fa = infr_ze::kernels::attn_fa().ok_or("infr-ze built without ocloc")?;
            let kq = gpu.module(kq).map_err(|e| e.to_string())?;
            let iq = gpu.module(iq).map_err(|e| e.to_string())?;
            let ew = gpu.module(ew).map_err(|e| e.to_string())?;
            let k16 = gpu.module(k16).map_err(|e| e.to_string())?;
            let fa = gpu.module(fa).map_err(|e| e.to_string())?;
            let kernel = |m: &std::rc::Rc<infr_ze::Module>, name: &str, wg: u32| {
                let k = m.kernel(name).map_err(|e| e.to_string())?;
                k.group_size(wg, 1, 1).map_err(|e| e.to_string())?;
                Ok::<_, String>(k)
            };
            let (q4k, q5k, iq4xs) = (
                kernel(&kq, "gemm_q4k", 128)?,
                kernel(&kq, "gemm_q5k", 128)?,
                kernel(&iq, "gemm_iq4xs", 128)?,
            );
            let (q6k, q3k, iq4nl) = (
                kernel(&k16, "gemm_q6k", 128)?,
                kernel(&k16, "gemm_q3k", 128)?,
                kernel(&iq, "gemm_iq4nl", 128)?,
            );
            let attn256 = kernel(&fa, "attn_fa256", 128)?;
            let (rmsnorm, swiglu, add, to_f16) = (
                kernel(&ew, "rmsnorm", 256)?,
                kernel(&ew, "swiglu", 256)?,
                kernel(&ew, "add", 256)?,
                kernel(&ew, "to_f16", 256)?,
            );
            let mut skinny = HashMap::new();
            for (dt, m, base) in [
                (DType::Q4K, &kq, "gemm_q4k"),
                (DType::Q5K, &kq, "gemm_q5k"),
                (DType::Iq4Xs, &iq, "gemm_iq4xs"),
                (DType::Q6K, &k16, "gemm_q6k"),
                (DType::Q3K, &k16, "gemm_q3k"),
                (DType::Iq4Nl, &iq, "gemm_iq4nl"),
            ] {
                skinny.insert((dt, false), kernel(m, &format!("{base}_m8"), 128)?);
                skinny.insert((dt, true), kernel(m, &format!("{base}_m16"), 128)?);
            }
            skinny.insert((DType::Q8_0, false), kernel(&iq, "gemm_q8_0_m8", 128)?);
            skinny.insert((DType::Q8_0, true), kernel(&iq, "gemm_q8_0_m16", 128)?);
            skinny.insert((DType::Q4_0, false), kernel(&iq, "gemm_q4_0_m8", 128)?);
            skinny.insert((DType::Q4_0, true), kernel(&iq, "gemm_q4_0_m16", 128)?);
            let dc = infr_ze::kernels::decode().ok_or("infr-ze built without ocloc")?;
            let ad = infr_ze::kernels::attn_dec().ok_or("infr-ze built without ocloc")?;
            let dc = gpu.module(dc).map_err(|e| e.to_string())?;
            let ad = gpu.module(ad).map_err(|e| e.to_string())?;
            let mut dec = HashMap::new();
            for (name, wg) in [
                ("copy_f32", 256),
                ("copy_strided", 256),
                ("store_f16", 256),
                ("copy_f16", 256),
                ("gated_act", 256),
                ("conv1d_silu", 256),
                ("deltanet_hist", 128),
                ("gated_rmsnorm", 128),
                ("qk_norm_rope", 64),
                ("block_conv", 256),
            ] {
                dec.insert(name, kernel(&dc, name, wg)?);
            }
            dec.insert("attn_dec256", kernel(&ad, "attn_dec256", 32)?);
            dec.insert("attn_dec_combine", kernel(&ad, "attn_dec_combine", 256)?);
            dec.insert("attn_ring128", kernel(&ad, "attn_ring128", 32)?);
            dec.insert("attn_combine_hd", kernel(&ad, "attn_combine_hd", 256)?);
            let sum_splits = kernel(&ew, "sum_splits", 256)?;
            let partials = gpu.alloc(SKINNY_PARTIALS).map_err(|e| e.to_string())?;
            let vk_sem = be_.tp_export_timeline().map_err(|e| e.to_string())?;
            let fd = unsafe { std::os::fd::BorrowedFd::borrow_raw(vk_sem.raw_fd()) }
                .try_clone_to_owned()
                .map_err(|e| e.to_string())?;
            let ze_sem = gpu.import_vk_timeline(fd).map_err(|e| e.to_string())?;
            Ok(ZeBridge(Mutex::new(Inner {
                gpu,
                q4k,
                q5k,
                iq4xs,
                q6k,
                q3k,
                iq4nl,
                attn256,
                rmsnorm,
                swiglu,
                add,
                to_f16,
                skinny,
                sum_splits,
                partials,
                dec,
                attn_parts: None,
                vk_sem,
                ze_sem,
                value: 0,
                imports: HashMap::new(),
            })))
        }

        /// The Vulkan timeline semaphore segment submits signal and wait.
        pub(crate) fn vk_semaphore(&self) -> vk::Semaphore {
            self.0.lock().unwrap().vk_sem.raw()
        }

        /// Two fresh timeline values: `v` for the Vulkan segment before the GEMM to signal, and
        /// `v + 1` (returned implicitly) for the GEMM to signal and the next segment to wait.
        pub(crate) fn reserve(&self) -> u64 {
            let mut g = self.0.lock().unwrap();
            g.value += 2;
            g.value - 1
        }

        /// Append a wait until the timeline reaches `v`: the start of a chain.
        pub(crate) fn wait(&self, v: u64) -> Result<()> {
            let g = self.0.lock().unwrap();
            g.gpu
                .wait_semaphore(&g.ze_sem, v)
                .map_err(|e| be(format!("ze wait: {e}")))
        }

        /// Append a signal of `v` after everything appended before: the end of a chain.
        pub(crate) fn signal(&self, v: u64) -> Result<()> {
            let g = self.0.lock().unwrap();
            g.gpu
                .signal_semaphore(&g.ze_sem, v)
                .map_err(|e| be(format!("ze signal: {e}")))
        }

        /// `C[m, out_f] = A16[m, in_f] · Wᵀ` on Level Zero. `a16`/`c` are exchange-arena views,
        /// `w` a weight in an exportable block with `w_off` an ELEMENT offset (row-aligned, as
        /// `Op::Linear`'s).
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn gemm(
            &self,
            be_: &VulkanBackend,
            dt: DType,
            a16: &dyn Buffer,
            w: &dyn Buffer,
            w_off: usize,
            c: &dyn Buffer,
            m: usize,
            in_f: usize,
            out_f: usize,
        ) -> Result<()> {
            let block = if m <= SKINNY_MAX_M {
                skinny_block(dt)
            } else {
                weight_block(dt)
            };
            let (vals, bytes, _) =
                block.ok_or_else(|| be(format!("ze gemm: no Level Zero kernel for {dt:?}")))?;
            let w_off_b = w_off / vals * bytes;
            let mut g = self.0.lock().unwrap();
            let (a_id, a_off) = g.import(be_, a16)?;
            let (w_id, w_base) = g.import(be_, w)?;
            let (c_id, c_off) = g.import(be_, c)?;
            let g = &*g;
            let e = |r: infr_ze::Result<()>| r.map_err(|e| be(format!("ze gemm: {e}")));
            if m <= SKINNY_MAX_M {
                // Skinny: one launch over all rows, K split into partials when N is narrow.
                let k = &g.skinny[&(dt, m > 8)];
                let nsplit = skinny_splits(dt, in_f, out_f);
                e(k.arg_buf_at(0, &g.imports[&a_id], a_off))?;
                e(k.arg_buf_at(1, &g.imports[&w_id], w_base + w_off_b))?;
                if nsplit > 1 {
                    e(k.arg_buf(2, &g.partials))?;
                } else {
                    e(k.arg_buf_at(2, &g.imports[&c_id], c_off))?;
                }
                e(k.arg(3, m as i32))?;
                e(k.arg(4, out_f as i32))?;
                e(k.arg(5, in_f as i32))?;
                e(k.arg(6, nsplit as i32))?;
                e(g.gpu.launch_labeled(
                    k,
                    [out_f.div_ceil(256) as u32, nsplit as u32, 1],
                    skinny_label(dt),
                ))?;
                if nsplit > 1 {
                    let s = &g.sum_splits;
                    e(s.arg_buf(0, &g.partials))?;
                    e(s.arg_buf_at(1, &g.imports[&c_id], c_off))?;
                    e(s.arg(2, (m * out_f) as i32))?;
                    e(s.arg(3, nsplit as i32))?;
                    e(g.gpu.launch_labeled(
                        s,
                        [(m * out_f).div_ceil(256) as u32, 1, 1],
                        "sum_splits",
                    ))?;
                }
                return Ok(());
            }
            let k = match dt {
                DType::Q4K => &g.q4k,
                DType::Q5K => &g.q5k,
                DType::Iq4Xs => &g.iq4xs,
                DType::Q6K => &g.q6k,
                DType::Q3K => &g.q3k,
                _ => &g.iq4nl,
            };
            e(k.arg_buf_at(1, &g.imports[&w_id], w_base + w_off_b))?;
            e(k.arg(4, out_f as i32))?;
            e(k.arg(5, in_f as i32))?;
            // One launch per 256-row packet (one workgroup row each): a packet's A stays in L2
            // across all the N tiles. Measured on the B70 (Q4_K): m=512 5120->17408 0.93 ms as one
            // launch vs 2 x 0.42 as packets, 17408->5120 1.07 vs 2 x 0.41 (85 -> 112 TFLOPS).
            // Arguments are captured at append, so each packet rebinds A, C and its row count.
            for r0 in (0..m).step_by(PACKET) {
                let mc = (m - r0).min(PACKET);
                e(k.arg_buf_at(0, &g.imports[&a_id], a_off + r0 * in_f * 2))?;
                e(k.arg_buf_at(2, &g.imports[&c_id], c_off + r0 * out_f * 4))?;
                e(k.arg(3, mc as i32))?;
                e(g.gpu
                    .launch_labeled(k, [1, (out_f / 64) as u32, 1], "gemm_prefill"))?;
            }
            Ok(())
        }

        /// `y = rmsnorm(x) * w` over `rows` rows of `dim` (and its f16 copy into `y16`).
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn rmsnorm(
            &self,
            be_: &VulkanBackend,
            x: &dyn Buffer,
            w: &dyn Buffer,
            y: &dyn Buffer,
            y16: Option<&dyn Buffer>,
            rows: usize,
            dim: usize,
            eps: f32,
        ) -> Result<()> {
            let mut g = self.0.lock().unwrap();
            let bufs = [x, w, y, y16.unwrap_or(y)];
            let ids = bufs
                .iter()
                .map(|b| g.import(be_, *b))
                .collect::<Result<Vec<_>>>()?;
            let g = &*g;
            let e = |r: infr_ze::Result<()>| r.map_err(|e| be(format!("ze rmsnorm: {e}")));
            for (i, (id, off)) in ids.iter().enumerate() {
                e(g.rmsnorm.arg_buf_at(i as u32, &g.imports[id], *off))?;
            }
            e(g.rmsnorm.arg(4, dim as i32))?;
            e(g.rmsnorm.arg(5, eps))?;
            e(g.rmsnorm.arg(6, i32::from(y16.is_some())))?;
            e(g.gpu
                .launch_labeled(&g.rmsnorm, [rows as u32, 1, 1], "rmsnorm"))
        }

        /// `y = silu(gate) * up` over `rows` rows of `[gate | up]` (`nff` each), with the optional
        /// SwiGLU clamp, and its f16 copy into `y16`.
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn swiglu(
            &self,
            be_: &VulkanBackend,
            gu: &dyn Buffer,
            y: &dyn Buffer,
            y16: Option<&dyn Buffer>,
            rows: usize,
            nff: usize,
            clamp: Option<f32>,
        ) -> Result<()> {
            let mut g = self.0.lock().unwrap();
            let bufs = [gu, y, y16.unwrap_or(y)];
            let ids = bufs
                .iter()
                .map(|b| g.import(be_, *b))
                .collect::<Result<Vec<_>>>()?;
            let g = &*g;
            let e = |r: infr_ze::Result<()>| r.map_err(|e| be(format!("ze swiglu: {e}")));
            for (i, (id, off)) in ids.iter().enumerate() {
                e(g.swiglu.arg_buf_at(i as u32, &g.imports[id], *off))?;
            }
            e(g.swiglu.arg(3, rows as i32))?;
            e(g.swiglu.arg(4, nff as i32))?;
            e(g.swiglu.arg(5, i32::from(clamp.is_some())))?;
            e(g.swiglu.arg(6, clamp.unwrap_or(0.0)))?;
            e(g.swiglu.arg(7, i32::from(y16.is_some())))?;
            e(g.gpu.launch_labeled(
                &g.swiglu,
                [(rows * nff).div_ceil(256) as u32, 1, 1],
                "swiglu",
            ))
        }

        /// Flash-attention prefill, head dim 256 (`attn_fa256`): `q` f16 `[rows, nh, 256]`, `kc`/`vc`
        /// f16 caches `[kv_len.., nkv, 256]` (no ring), `o` f32 `[rows, nh, 256]`; causal from
        /// `pos`, `window` (0 = none), `scale` (0 = 1/16).
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn attention(
            &self,
            be_: &VulkanBackend,
            q: &dyn Buffer,
            kc: &dyn Buffer,
            vc: &dyn Buffer,
            o: &dyn Buffer,
            rows: usize,
            kv_len: usize,
            nh: usize,
            nkv: usize,
            pos: usize,
            window: usize,
            scale: f32,
        ) -> Result<()> {
            let mut g = self.0.lock().unwrap();
            let ids = [q, kc, vc, o]
                .iter()
                .map(|b| g.import(be_, *b))
                .collect::<Result<Vec<_>>>()?;
            let g = &*g;
            let k = &g.attn256;
            let e = |r: infr_ze::Result<()>| r.map_err(|e| be(format!("ze attention: {e}")));
            for (i, (id, off)) in ids.iter().enumerate() {
                e(k.arg_buf_at(i as u32, &g.imports[id], *off))?;
            }
            for (i, v) in [rows, kv_len, nh, nkv, pos, window].iter().enumerate() {
                e(k.arg(4 + i as u32, *v as i32))?;
            }
            e(k.arg(10, scale))?;
            e(g.gpu
                .launch_labeled(k, [(rows.div_ceil(32) * nh) as u32, 1, 1], "attn_fa256"))
        }

        /// `dst = a + b` over `n` floats.
        pub(crate) fn add(
            &self,
            be_: &VulkanBackend,
            a: &dyn Buffer,
            b: &dyn Buffer,
            dst: &dyn Buffer,
            n: usize,
        ) -> Result<()> {
            let mut g = self.0.lock().unwrap();
            let ids = [a, b, dst]
                .iter()
                .map(|x| g.import(be_, *x))
                .collect::<Result<Vec<_>>>()?;
            let g = &*g;
            let e = |r: infr_ze::Result<()>| r.map_err(|e| be(format!("ze add: {e}")));
            for (i, (id, off)) in ids.iter().enumerate() {
                e(g.add.arg_buf_at(i as u32, &g.imports[id], *off))?;
            }
            e(g.add.arg(3, n as i32))?;
            e(g.gpu
                .launch_labeled(&g.add, [n.div_ceil(256) as u32, 1, 1], "add"))
        }

        /// `y16 = (half) x` over `n` floats — a GEMM's A copy no producer wrote.
        pub(crate) fn to_f16(
            &self,
            be_: &VulkanBackend,
            x: &dyn Buffer,
            y16: &dyn Buffer,
            n: usize,
        ) -> Result<()> {
            let mut g = self.0.lock().unwrap();
            let ids = [x, y16]
                .iter()
                .map(|b| g.import(be_, *b))
                .collect::<Result<Vec<_>>>()?;
            let g = &*g;
            let e = |r: infr_ze::Result<()>| r.map_err(|e| be(format!("ze to_f16: {e}")));
            for (i, (id, off)) in ids.iter().enumerate() {
                e(g.to_f16.arg_buf_at(i as u32, &g.imports[id], *off))?;
            }
            e(g.to_f16.arg(2, n as i32))?;
            e(g.gpu
                .launch_labeled(&g.to_f16, [n.div_ceil(256) as u32, 1, 1], "to_f16"))
        }

        /// Launch the verify-sized kernel `name` (`decode.cl`) with `args` in order — buffers are
        /// imported (exportable allocations, at their offsets), ints and floats passed as values.
        pub(crate) fn run(
            &self,
            be_: &VulkanBackend,
            name: &'static str,
            args: &[ZArg],
            groups: [u32; 3],
        ) -> Result<()> {
            let mut g = self.0.lock().unwrap();
            let mut ids = Vec::with_capacity(args.len());
            for a in args {
                ids.push(match a {
                    ZArg::B(b) | ZArg::O(b, _) => Some(g.import(be_, *b)?),
                    _ => None,
                });
            }
            let g = &*g;
            let k = g
                .dec
                .get(name)
                .ok_or_else(|| be(format!("ze: no kernel {name}")))?;
            let e = |r: infr_ze::Result<()>| r.map_err(|e| be(format!("ze {name}: {e}")));
            for (i, (a, id)) in args.iter().zip(&ids).enumerate() {
                let i = i as u32;
                match (a, id) {
                    (ZArg::B(_), Some((id, off))) => e(k.arg_buf_at(i, &g.imports[id], *off))?,
                    (ZArg::O(_, extra), Some((id, off))) => {
                        e(k.arg_buf_at(i, &g.imports[id], *off + *extra))?
                    }
                    (ZArg::I(v), _) => e(k.arg(i, *v))?,
                    (ZArg::F(v), _) => e(k.arg(i, *v))?,
                    (ZArg::B(_) | ZArg::O(..), None) => unreachable!("imported above"),
                }
            }
            e(g.gpu.launch_labeled(k, groups, name))
        }

        /// Verify-sized attention, head dim 256 (`attn_dec256` + `attn_dec_combine`): `q` f16
        /// `[rows, nh, 256]`, `kc`/`vc` f16 caches (no ring), `o` f32 `[rows, nh, 256]`; causal
        /// from `pos`, `scale` (0 = 1/16). The key range is split into chunks so a long context
        /// fills the card.
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn attention_dec(
            &self,
            be_: &VulkanBackend,
            q: &dyn Buffer,
            kc: &dyn Buffer,
            vc: &dyn Buffer,
            o: &dyn Buffer,
            rows: usize,
            kv_len: usize,
            nh: usize,
            nkv: usize,
            pos: usize,
            scale: f32,
        ) -> Result<()> {
            // ~32 chunks of a multiple of the 32-key block, 64..=2048 keys each.
            let chunk = kv_len.div_ceil(32).next_multiple_of(32).clamp(64, 2048);
            let nchunk = kv_len.div_ceil(chunk).max(1);
            let units = nchunk * rows * nh;
            let mut g = self.0.lock().unwrap();
            if g.attn_parts.as_ref().is_none_or(|(_, _, cap)| *cap < units) {
                let cap = units.next_power_of_two();
                let po = g
                    .gpu
                    .alloc(cap * 256 * 4)
                    .map_err(|e| be(format!("ze alloc: {e}")))?;
                let pml = g
                    .gpu
                    .alloc(cap * 2 * 4)
                    .map_err(|e| be(format!("ze alloc: {e}")))?;
                g.attn_parts = Some((po, pml, cap));
            }
            let ids = [q, kc, vc, o]
                .iter()
                .map(|b| g.import(be_, *b))
                .collect::<Result<Vec<_>>>()?;
            let g = &*g;
            let (po, pml, _) = g.attn_parts.as_ref().expect("allocated above");
            let e = |r: infr_ze::Result<()>| r.map_err(|e| be(format!("ze attention_dec: {e}")));
            let k = &g.dec["attn_dec256"];
            for (i, (id, off)) in ids[..3].iter().enumerate() {
                e(k.arg_buf_at(i as u32, &g.imports[id], *off))?;
            }
            e(k.arg_buf(3, po))?;
            e(k.arg_buf(4, pml))?;
            for (i, v) in [rows, kv_len, nh, nkv, pos, chunk].iter().enumerate() {
                e(k.arg(5 + i as u32, *v as i32))?;
            }
            e(k.arg(11, scale))?;
            e(g.gpu.launch_labeled(
                k,
                [nh as u32, nchunk as u32, rows.div_ceil(8) as u32],
                "attn_dec256",
            ))?;
            let c = &g.dec["attn_dec_combine"];
            e(c.arg_buf(0, po))?;
            e(c.arg_buf(1, pml))?;
            let (oid, ooff) = ids[3];
            e(c.arg_buf_at(2, &g.imports[&oid], ooff))?;
            e(c.arg(3, rows as i32))?;
            e(c.arg(4, nh as i32))?;
            e(c.arg(5, nchunk as i32))?;
            e(g.gpu
                .launch_labeled(c, [(rows * nh) as u32, 1, 1], "attn_dec_combine"))
        }

        /// Open a Level Zero per-kernel profile (`INFR_PROF_OPS`) for the forward about to run.
        pub(crate) fn profile_begin(&self) -> Result<()> {
            let g = self.0.lock().unwrap();
            g.gpu
                .profile_begin(16384)
                .map_err(|e| be(format!("ze profile: {e}")))
        }

        /// Close the profile and fold it, by label, into the `[prof:ze]` report.
        pub(crate) fn profile_report(&self) -> Result<()> {
            let g = self.0.lock().unwrap();
            let rows = g
                .gpu
                .profile_take()
                .map_err(|e| be(format!("ze profile: {e}")))?;
            let mut p = infr_core::prof::OpProf::new("ze", infr_core::prof::Unit::Device);
            for (label, ns) in rows {
                p.add(label, ns / 1e3);
            }
            p.flush();
            Ok(())
        }

        /// The DFlash drafter's block attention (`attn_ring128` + `attn_combine_hd`): head dim 128,
        /// every row sees keys `[lo, kv_len)`, K/V caches a ring of `cap` rows (cap % 32 == 0).
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn attention_ring(
            &self,
            be_: &VulkanBackend,
            q: &dyn Buffer,
            kc: &dyn Buffer,
            vc: &dyn Buffer,
            o: &dyn Buffer,
            rows: usize,
            kv_len: usize,
            nh: usize,
            nkv: usize,
            lo: usize,
            cap: usize,
            scale: f32,
        ) -> Result<()> {
            let span = kv_len - (lo & !31);
            let chunk = span.div_ceil(16).next_multiple_of(32).clamp(64, 1024);
            let nchunk = span.div_ceil(chunk).max(1);
            let units = nchunk * rows * nh;
            let mut g = self.0.lock().unwrap();
            if g.attn_parts.as_ref().is_none_or(|(_, _, cap)| *cap < units) {
                let c = units.next_power_of_two();
                let po = g
                    .gpu
                    .alloc(c * 256 * 4)
                    .map_err(|e| be(format!("ze alloc: {e}")))?;
                let pml = g
                    .gpu
                    .alloc(c * 2 * 4)
                    .map_err(|e| be(format!("ze alloc: {e}")))?;
                g.attn_parts = Some((po, pml, c));
            }
            let ids = [q, kc, vc, o]
                .iter()
                .map(|b| g.import(be_, *b))
                .collect::<Result<Vec<_>>>()?;
            let g = &*g;
            let (po, pml, _) = g.attn_parts.as_ref().expect("allocated above");
            let e = |r: infr_ze::Result<()>| r.map_err(|e| be(format!("ze attention_ring: {e}")));
            let k = &g.dec["attn_ring128"];
            for (i, (id, off)) in ids[..3].iter().enumerate() {
                e(k.arg_buf_at(i as u32, &g.imports[id], *off))?;
            }
            e(k.arg_buf(3, po))?;
            e(k.arg_buf(4, pml))?;
            for (i, v) in [rows, kv_len, nh, nkv, lo, cap, chunk].iter().enumerate() {
                e(k.arg(5 + i as u32, *v as i32))?;
            }
            e(k.arg(12, scale))?;
            e(g.gpu.launch_labeled(
                k,
                [nh as u32, nchunk as u32, rows.div_ceil(8) as u32],
                "attn_ring128",
            ))?;
            let c = &g.dec["attn_combine_hd"];
            e(c.arg_buf(0, po))?;
            e(c.arg_buf(1, pml))?;
            let (oid, ooff) = ids[3];
            e(c.arg_buf_at(2, &g.imports[&oid], ooff))?;
            e(c.arg(3, rows as i32))?;
            e(c.arg(4, nh as i32))?;
            e(c.arg(5, nchunk as i32))?;
            e(c.arg(6, 128i32))?;
            e(g.gpu
                .launch_labeled(c, [(rows * nh) as u32, 1, 1], "attn_combine_hd"))
        }

        /// Drop the import of a retired exportable allocation.
        fn forget(&self, export_id: u64) {
            self.0.lock().unwrap().imports.remove(&export_id);
        }
    }

    impl Inner {
        /// The Level Zero import holding `buf` (importing its block on first use) and `buf`'s byte
        /// offset in it.
        fn import(&mut self, be_: &VulkanBackend, buf: &dyn Buffer) -> Result<(u64, usize)> {
            let vb = as_vk_buf(buf)?;
            let (block, off): (&VkBuffer, usize) = match &vb.backing {
                Backing::BdaSub(h) => (&h.buf, vb.sub_offset),
                Backing::External { .. } => (vb, 0),
                _ => return Err(be("ze gemm: operand is not in an exportable allocation")),
            };
            let Backing::External {
                memory, export_id, ..
            } = &block.backing
            else {
                return Err(be("ze gemm: operand's block is not exportable"));
            };
            if *export_id == 0 {
                return Err(be("ze gemm: operand's block is not a same-card export"));
            }
            if !self.imports.contains_key(export_id) {
                let fd = be_.export_dmabuf(*memory)?;
                let dev = self
                    .gpu
                    .import_dmabuf(fd, block.mem_size as usize)
                    .map_err(|e| be(format!("ze import: {e}")))?;
                self.imports.insert(*export_id, dev);
            }
            Ok((*export_id, off))
        }
    }

    impl VulkanBackend {
        /// Does this backend route prefill kernels to Level Zero? Intel Xe2 only, and only
        /// with the external memory/semaphore extensions the hand-off needs.
        pub(crate) fn ze_wanted(&self) -> bool {
            self.cfg().kernels.vulkan.ze_gemm
                && self.caps().f16_coopmat_8x16()
                && self.p2p_supported(crate::P2pHandleType::DmaBuf)
                && self.external_semaphore_supported()
                && self.shared.pci_bdf.is_some()
        }

        /// Is `buf` in a same-card exportable allocation Level Zero can import (directly, or as a
        /// view of one)?
        pub(crate) fn ze_operand_ok(&self, buf: &dyn Buffer) -> bool {
            let Ok(vb) = as_vk_buf(buf) else {
                return false;
            };
            let block = match &vb.backing {
                Backing::BdaSub(h) => &h.buf,
                _ => vb,
            };
            matches!(block.backing, Backing::External { export_id, .. } if export_id != 0)
        }

        /// The bridge, opened on first use; `None` when not wanted or Level Zero is unavailable
        /// (reported once on stderr).
        pub(crate) fn ze_bridge(&self) -> Option<&ZeBridge> {
            self.ze
                .bridge
                .get_or_init(|| {
                    if !self.ze_wanted() {
                        return None;
                    }
                    match ZeBridge::open(self) {
                        Ok(b) => Some(b),
                        Err(e) => {
                            tracing::warn!(
                                "[infr] Level Zero unavailable ({e}): prefill stays on Vulkan (install the Intel compute runtime for the faster Xe2 prefill, or set INFR_ZE_GEMM=0 to silence this)"
                            );
                            None
                        }
                    }
                })
                .as_ref()
        }

        /// Start a forward's exchange-arena use: at least `bytes` free from offset 0. Views from
        /// the previous forward must be dead (one forward at a time per backend). A too-small
        /// arena is replaced — its Level Zero import is dropped, its memory freed with its last view.
        pub(crate) fn ze_arena_reset(&self, bytes: usize) -> Result<()> {
            let mut arena = self.ze.arena.lock().unwrap();
            if let Some(a) = arena.as_mut() {
                if a.size >= bytes {
                    a.cursor = 0;
                    return Ok(());
                }
            }
            if let Some(old) = arena.take() {
                if let Some(b) = self.ze_bridge() {
                    b.forget(old.export_id);
                }
            }
            let size = bytes.next_multiple_of(64 << 20);
            let mut buf = self.alloc_exportable(size)?;
            self.track_activation(&mut buf);
            let export_id = match &buf.backing {
                Backing::External { export_id, .. } => *export_id,
                _ => unreachable!("alloc_exportable returns an External buffer"),
            };
            let base_addr = buf.own_addr.expect("alloc_exportable sets own_addr");
            *arena = Some(Arena {
                handle: Arc::new(BdaBlockHandle { buf, base_addr }),
                size,
                cursor: 0,
                export_id,
            });
            Ok(())
        }

        /// A `bytes`-long view into the exchange arena (256-byte aligned) — a graph tensor or
        /// scratch buffer Level Zero can address.
        pub(crate) fn ze_view(&self, bytes: usize) -> Result<Box<dyn Buffer>> {
            let mut arena = self.ze.arena.lock().unwrap();
            let a = arena
                .as_mut()
                .ok_or_else(|| be("ze_view before ze_arena_reset"))?;
            let off = a.cursor.next_multiple_of(256);
            if off + bytes > a.size {
                return Err(be(format!(
                    "ze exchange arena overflow: {} + {bytes} > {}",
                    off, a.size
                )));
            }
            a.cursor = off + bytes;
            Ok(Box::new(VkBuffer {
                shared: Arc::clone(&self.shared),
                buffer: a.handle.buf.buffer,
                backing: Backing::BdaSub(Arc::clone(&a.handle)),
                size: bytes,
                mem_size: 0,
                location: MemoryLocation::GpuOnly,
                sub_offset: off,
                own_addr: None,
                act_bytes: 0,
            }))
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod imp {
    use super::*;

    #[derive(Default)]
    pub(crate) struct ZeCell;

    /// Never constructed off Linux.
    pub(crate) enum ZeBridge {}

    #[allow(clippy::too_many_arguments)]
    impl ZeBridge {
        pub(crate) fn vk_semaphore(&self) -> ash::vk::Semaphore {
            match *self {}
        }
        pub(crate) fn reserve(&self) -> u64 {
            match *self {}
        }
        pub(crate) fn wait(&self, _: u64) -> Result<()> {
            match *self {}
        }
        pub(crate) fn signal(&self, _: u64) -> Result<()> {
            match *self {}
        }
        pub(crate) fn gemm(
            &self,
            _: &VulkanBackend,
            _: DType,
            _: &dyn Buffer,
            _: &dyn Buffer,
            _: usize,
            _: &dyn Buffer,
            _: usize,
            _: usize,
            _: usize,
        ) -> Result<()> {
            match *self {}
        }
        pub(crate) fn rmsnorm(
            &self,
            _: &VulkanBackend,
            _: &dyn Buffer,
            _: &dyn Buffer,
            _: &dyn Buffer,
            _: Option<&dyn Buffer>,
            _: usize,
            _: usize,
            _: f32,
        ) -> Result<()> {
            match *self {}
        }
        pub(crate) fn swiglu(
            &self,
            _: &VulkanBackend,
            _: &dyn Buffer,
            _: &dyn Buffer,
            _: Option<&dyn Buffer>,
            _: usize,
            _: usize,
            _: Option<f32>,
        ) -> Result<()> {
            match *self {}
        }
        pub(crate) fn add(
            &self,
            _: &VulkanBackend,
            _: &dyn Buffer,
            _: &dyn Buffer,
            _: &dyn Buffer,
            _: usize,
        ) -> Result<()> {
            match *self {}
        }
        pub(crate) fn attention(
            &self,
            _: &VulkanBackend,
            _: &dyn Buffer,
            _: &dyn Buffer,
            _: &dyn Buffer,
            _: &dyn Buffer,
            _: usize,
            _: usize,
            _: usize,
            _: usize,
            _: usize,
            _: usize,
            _: f32,
        ) -> Result<()> {
            match *self {}
        }
        pub(crate) fn to_f16(
            &self,
            _: &VulkanBackend,
            _: &dyn Buffer,
            _: &dyn Buffer,
            _: usize,
        ) -> Result<()> {
            match *self {}
        }
        pub(crate) fn run(
            &self,
            _: &VulkanBackend,
            _: &'static str,
            _: &[super::ZArg],
            _: [u32; 3],
        ) -> Result<()> {
            match *self {}
        }
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn attention_dec(
            &self,
            _: &VulkanBackend,
            _: &dyn Buffer,
            _: &dyn Buffer,
            _: &dyn Buffer,
            _: &dyn Buffer,
            _: usize,
            _: usize,
            _: usize,
            _: usize,
            _: usize,
            _: f32,
        ) -> Result<()> {
            match *self {}
        }
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn attention_ring(
            &self,
            _: &VulkanBackend,
            _: &dyn Buffer,
            _: &dyn Buffer,
            _: &dyn Buffer,
            _: &dyn Buffer,
            _: usize,
            _: usize,
            _: usize,
            _: usize,
            _: usize,
            _: usize,
            _: f32,
        ) -> Result<()> {
            match *self {}
        }
        pub(crate) fn profile_begin(&self) -> Result<()> {
            match *self {}
        }
        pub(crate) fn profile_report(&self) -> Result<()> {
            match *self {}
        }
    }

    impl VulkanBackend {
        pub(crate) fn ze_wanted(&self) -> bool {
            false
        }
        pub(crate) fn ze_bridge(&self) -> Option<&ZeBridge> {
            None
        }
        pub(crate) fn ze_operand_ok(&self, _: &dyn Buffer) -> bool {
            false
        }
        pub(crate) fn ze_arena_reset(&self, _: usize) -> Result<()> {
            Ok(())
        }
        pub(crate) fn ze_view(&self, _: usize) -> Result<Box<dyn Buffer>> {
            Err(crate::be("ze_view: Level Zero GEMMs are Linux-only"))
        }
    }
}

pub(crate) use imp::ZeCell;
