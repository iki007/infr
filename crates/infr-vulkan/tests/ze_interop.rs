//! Vulkan ↔ Level Zero on ONE Intel GPU (backlog B102, the hybrid prefill path): Vulkan buffers
//! exported as dma-buf and imported into Level Zero, and a Vulkan timeline semaphore imported into
//! Level Zero, so prefill GEMMs can run as OpenCL C DPAS kernels between Vulkan dispatches.
//!
//! Hardware-gated (`#[ignore]`): needs a GPU that both Mesa ANV and the Intel compute runtime
//! drive. `INFR_ZE_VK` picks the Vulkan device index (default 1) and `INFR_ZE_PCI` the same card's
//! Level Zero device (default `03:00.0`) — on the B70 box, the headless card.
#![cfg(target_os = "linux")]

use infr_core::backend::{Backend, BufferUsage};
use infr_vulkan::{P2pHandleType, VulkanBackend};
use std::time::{Duration, Instant};

fn as_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding or invalid bit patterns as bytes.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn floats(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|w| f32::from_le_bytes(*w))
        .collect()
}

struct Rig {
    vk: VulkanBackend,
    g: infr_ze::Gpu,
    bin: &'static [u8],
}

fn rig() -> Option<Rig> {
    let idx = std::env::var("INFR_ZE_VK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let pci = std::env::var("INFR_ZE_PCI").unwrap_or_else(|_| "03:00.0".into());
    let vk = VulkanBackend::new_on(idx).ok()?;
    if !vk.p2p_supported(P2pHandleType::DmaBuf) {
        eprintln!("skip: Vulkan device {idx} has no dma-buf export");
        return None;
    }
    let ze = infr_ze::Ze::open().ok()?;
    let g = infr_ze::Gpu::open(&ze, Some(&pci)).ok()?;
    let bin = infr_ze::kernels::vadd()?;
    println!(
        "Vulkan device {idx} ({}) <-> Level Zero {pci} ({})",
        vk.name(),
        g.info.name
    );
    Some(Rig { vk, g, bin })
}

/// A duplicate for an importer that takes ownership (Level Zero closes the fd it imports).
fn dup(fd: i32) -> std::os::fd::OwnedFd {
    // SAFETY: `fd` is an open descriptor owned by a live export for the duration of the call.
    unsafe { std::os::fd::BorrowedFd::borrow_raw(fd) }
        .try_clone_to_owned()
        .expect("dup")
}

fn fd_open(fd: i32) -> bool {
    // SAFETY: F_GETFD only queries the descriptor table.
    unsafe { libc::fcntl(fd, libc::F_GETFD) >= 0 }
}

/// Vulkan writes → Level Zero reads and writes → Vulkan reads, over several rounds (stale caches
/// or a wrong alias would show as a mismatch in a later round), ordered by host fences.
#[test]
#[ignore = "needs an Intel GPU with both Mesa ANV and Level Zero"]
fn dmabuf_round_trip_vulkan_level_zero() {
    let Some(Rig { vk, g, bin }) = rig() else {
        eprintln!("skip: no Vulkan+Level Zero device pair or no ocloc kernels");
        return;
    };
    let n = 1usize << 20;
    let bytes = n * 4;
    let ea = vk.p2p_export(bytes, P2pHandleType::DmaBuf).unwrap();
    let ec = vk.p2p_export(bytes, P2pHandleType::DmaBuf).unwrap();
    let za = g
        .import_dmabuf(dup(ea.raw_fd()), bytes)
        .expect("Level Zero dma-buf import");
    let zc = g.import_dmabuf(dup(ec.raw_fd()), bytes).unwrap();
    assert!(
        fd_open(ea.raw_fd()) && fd_open(ec.raw_fd()),
        "the export lost its own fd"
    );
    let b: Vec<f32> = (0..n).map(|i| (i % 977) as f32 * 0.25).collect();
    let zb = g.alloc(bytes).unwrap();
    g.upload(&zb, as_bytes(&b)).unwrap();
    let m = g.module(bin).unwrap();
    let k = m.kernel("vadd").unwrap();
    k.group_size(256, 1, 1).unwrap();
    k.arg_buf(0, &za).unwrap();
    k.arg_buf(1, &zb).unwrap();
    k.arg_buf(2, &zc).unwrap();
    k.arg(3, n as i32).unwrap();
    for round in 0..4u32 {
        let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 + round as f32).collect();
        vk.upload(ea.buffer(), as_bytes(&a)).unwrap();
        vk.sync().unwrap();
        g.launch(&k, [(n / 256) as u32, 1, 1]).unwrap();
        g.sync().unwrap();
        let mut out = vec![0u8; bytes];
        vk.download(ec.buffer(), &mut out).unwrap();
        let c = floats(&out);
        for i in 0..n {
            assert_eq!(c[i], a[i] + b[i], "round {round}, element {i}");
        }
    }
    // And the other direction for the input: Level Zero overwrites `za`, Vulkan reads it.
    g.upload(&za, as_bytes(&b)).unwrap();
    let mut out = vec![0u8; bytes];
    vk.download(ea.buffer(), &mut out).unwrap();
    assert_eq!(floats(&out), b, "Vulkan reads what Level Zero wrote");
    println!("dma-buf round trip: 4 rounds x {n} floats exact, both directions");
}

/// Level Zero waits on and signals a Vulkan timeline semaphore; then the per-handoff cost of a
/// Vulkan → Level Zero → Vulkan chain, ordered by host fences against GPU-side semaphores.
#[test]
#[ignore = "needs an Intel GPU with both Mesa ANV and Level Zero; prints handoff costs"]
fn timeline_semaphore_handoff() {
    let Some(Rig { vk, g, bin }) = rig() else {
        eprintln!("skip: no Vulkan+Level Zero device pair or no ocloc kernels");
        return;
    };
    let n = 4096usize;
    let bytes = n * 4;
    let ea = vk.p2p_export(bytes, P2pHandleType::DmaBuf).unwrap();
    let ec = vk.p2p_export(bytes, P2pHandleType::DmaBuf).unwrap();
    let za = g.import_dmabuf(dup(ea.raw_fd()), bytes).unwrap();
    let zc = g.import_dmabuf(dup(ec.raw_fd()), bytes).unwrap();
    let zb = g.alloc(bytes).unwrap();
    let b: Vec<f32> = (0..n).map(|i| i as f32).collect();
    g.upload(&zb, as_bytes(&b)).unwrap();
    let m = g.module(bin).unwrap();
    let k = m.kernel("vadd").unwrap();
    k.group_size(256, 1, 1).unwrap();
    k.arg_buf(0, &za).unwrap();
    k.arg_buf(1, &zb).unwrap();
    k.arg_buf(2, &zc).unwrap();
    k.arg(3, n as i32).unwrap();
    let groups = [(n / 256) as u32, 1, 1];

    let sem = vk.tp_export_timeline().expect("Vulkan timeline export");
    let zs = match g.import_vk_timeline(dup(sem.raw_fd())) {
        Ok(s) => s,
        Err(e) => {
            println!("Level Zero rejected the Vulkan timeline semaphore: {e}");
            return;
        }
    };
    assert!(fd_open(sem.raw_fd()), "the export lost its own fd");

    // Ordering: Level Zero waits for 1, adds, signals 2. The host publishes new input only then
    // signals 1 — the kernel must see that input, not the earlier one.
    let a0 = vec![-1.0f32; n];
    vk.upload(ea.buffer(), as_bytes(&a0)).unwrap();
    vk.sync().unwrap();
    let t = Instant::now();
    g.wait_semaphore(&zs, 1).unwrap();
    let append_wait = t.elapsed();
    g.launch(&k, groups).unwrap();
    g.signal_semaphore(&zs, 2).unwrap();
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(
        sem.counter_value().unwrap(),
        0,
        "the kernel ran before the wait was met"
    );
    let a1: Vec<f32> = (0..n).map(|i| 0.5 * i as f32).collect();
    vk.upload(ea.buffer(), as_bytes(&a1)).unwrap();
    vk.sync().unwrap();
    sem.host_signal(1).unwrap();
    assert!(
        sem.host_wait(2, 5_000_000_000).unwrap(),
        "Level Zero never signalled 2"
    );
    let mut out = vec![0u8; bytes];
    vk.download(ec.buffer(), &mut out).unwrap();
    let c = floats(&out);
    for i in 0..n {
        assert_eq!(c[i], a1[i] + b[i], "element {i}");
    }
    println!(
        "Level Zero waited on / signalled the Vulkan timeline (append of an unmet wait took {:.1} us)",
        append_wait.as_secs_f64() * 1e6
    );

    // Per-handoff cost. One round = Vulkan copy into `ea` → Level Zero add → Vulkan copy out of
    // `ec`: two cross-API handoffs around a ~µs kernel.
    let src = vk.alloc_uninit(bytes, BufferUsage::Weights).unwrap();
    let dst = vk.alloc_uninit(bytes, BufferUsage::Weights).unwrap();
    vk.upload(src.as_ref(), as_bytes(&a1)).unwrap();
    vk.sync().unwrap();
    let rounds = 200u64;

    let t = Instant::now();
    for _ in 0..rounds {
        vk.p2p_publish_copy(src.as_ref(), ea.buffer(), bytes)
            .unwrap();
        g.launch(&k, groups).unwrap();
        g.sync().unwrap();
        vk.p2p_gather_copy(ec.buffer(), dst.as_ref(), bytes)
            .unwrap();
    }
    let host = t.elapsed().as_secs_f64() / rounds as f64;

    // The parts alone: Level Zero launches back to back, and Vulkan's two copies per round as
    // semaphore-free submits on its own queue.
    let t = Instant::now();
    for _ in 0..rounds {
        g.launch(&k, groups).unwrap();
    }
    g.sync().unwrap();
    let ze_only = t.elapsed().as_secs_f64() / rounds as f64;
    let t = Instant::now();
    for _ in 0..rounds {
        vk.copy_buffer(src.as_ref(), ea.buffer(), bytes).unwrap();
        vk.copy_buffer(ec.buffer(), dst.as_ref(), bytes).unwrap();
    }
    vk.sync().unwrap();
    let vk_only = t.elapsed().as_secs_f64() / rounds as f64;
    println!(
        "alone: Level Zero launch {:.1} us, Vulkan 2 copies {:.1} us (per round)",
        ze_only * 1e6,
        vk_only * 1e6
    );

    let imp = vk
        .tp_import_timeline(&sem)
        .expect("same-device timeline import");
    let base = 2u64;
    let t = Instant::now();
    let mut cmds = Vec::new();
    for r in 0..rounds {
        let v = base + 2 * r;
        cmds.push(
            vk.tp_submit_copy_signal(src.as_ref(), ea.buffer(), bytes, &sem, v + 1)
                .unwrap(),
        );
        g.wait_semaphore(&zs, v + 1).unwrap();
        g.launch(&k, groups).unwrap();
        g.signal_semaphore(&zs, v + 2).unwrap();
        cmds.push(
            vk.tp_submit_copies_wait(&[(ec.buffer(), dst.as_ref(), bytes)], &[(&imp, v + 2)])
                .unwrap(),
        );
    }
    let issued = t.elapsed().as_secs_f64() / rounds as f64;
    vk.tp_queue_wait_idle().unwrap();
    let gpu = t.elapsed().as_secs_f64() / rounds as f64;
    vk.tp_free_cmds(&cmds);

    // The same two Vulkan submits per round, chained by the timeline with no Level Zero between.
    let base = base + 2 * rounds;
    let t = Instant::now();
    let mut cmds = Vec::new();
    for r in 0..rounds {
        let v = base + 2 * r;
        cmds.push(
            vk.tp_submit_copy_signal(src.as_ref(), ea.buffer(), bytes, &sem, v + 1)
                .unwrap(),
        );
        cmds.push(
            vk.tp_submit_copies_wait(&[(ec.buffer(), dst.as_ref(), bytes)], &[(&imp, v + 1)])
                .unwrap(),
        );
    }
    vk.tp_queue_wait_idle().unwrap();
    let vk_chain = t.elapsed().as_secs_f64() / rounds as f64;
    vk.tp_free_cmds(&cmds);
    println!(
        "Vulkan-only chain of the same 2 submits: {:.1} us per round",
        vk_chain * 1e6
    );
    let mut out = vec![0u8; bytes];
    vk.download(dst.as_ref(), &mut out).unwrap();
    let c = floats(&out);
    for i in 0..n {
        assert_eq!(c[i], a1[i] + b[i], "semaphore chain element {i}");
    }
    println!(
        "round (Vulkan copy -> L0 add -> Vulkan copy): host fences {:.1} us, timeline semaphores \
         {:.1} us ({:.1} us of it host issue)",
        host * 1e6,
        gpu * 1e6,
        issued * 1e6
    );
}
