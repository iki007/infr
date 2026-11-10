# Level Zero backend (`infr-ze`) — Intel Xe2, native

Status: **phase 0 and the interop phase done** (2026-11-10). The device layer, a DPAS GEMM probe and
Vulkan ↔ Level Zero memory/semaphore sharing run on the Arc Pro B70; nothing in the engine uses the
crate yet. Backlog B102.

## Why

On the B70, the Vulkan prefill GEMM (`native_gemm_warp` `_x256`, Mesa ANV cooperative matrices)
peaks at ~50 TFLOPS and sustains ~46 (`docs/intel-b70.md`); llama.cpp's Vulkan GEMM holds ~57, and
llama.cpp SYCL (oneDNN) is 1.5–1.7x our prefill. TensorFold's Level Zero port, on the same card,
sustains 146 TFLOPS fp16 on the gate|up GEMM with nGEN-generated ISA. The question for phase 0 was
whether infr can get there without C++: OpenCL C compiled offline, driven through Level Zero.

## Shape

- **Level Zero through `dlopen`** (`crates/infr-ze/src/ffi.rs`, the `ash` analogue). The C structs
  are hand-declared from `ze_api.h` (loader 1.32) and the entry points resolved from
  `libze_loader.so.1` / `ze_loader.dll` at run time. Nothing links against oneAPI.
- **Kernels in OpenCL C, compiled by `ocloc`** at build time into native binaries embedded in the
  crate (`build.rs`, the `glslc` analogue; target `bmg-g31`, `INFR_ZE_TARGET` overrides). A kernel's
  first lines can carry `// ocloc-options:` (e.g. `-cl-intel-256-GRF-per-thread`). Without `ocloc`
  the crate builds with no kernels and the probes skip — CI and non-Intel machines are unaffected.
- OpenCL C reaches the Xe2 primitives the Vulkan path cannot: **DPAS**
  (`intel_sub_group_f16_f16_matrix_mad_k16`, 16-lane subgroups) and **2D block loads/stores**
  (`intel_sub_group_2d_block_read_*`, including the transposed and VNNI-transform forms). The
  probe's ISA (`ocloc disasm`) shows `dpas.8x8 (16|M0)`, `load_block2d.ugm.d16`/`.d16v` and
  `store_block2d`.
- The device layer (`crates/infr-ze/src/lib.rs`): device discovery by PCI address (the box has
  two identical B70s; `INFR_ZE_PCI=03:00.0` is the headless one), a context with one in-order
  immediate command list, device memory, modules, kernels, kernel timestamps.

## Phase 0 results (Arc Pro B70 `03:00.0`, compute runtime 26.40, `tests/probe.rs`)

`kernels/gemm_f16.cl`: `C[M,N] (f32) = A[M,K] (f16) · W[N,K]ᵀ (f16)`, W in weight layout.
Each 16-lane subgroup keeps a 32x64 C tile in registers (4x4 DPAS accumulators, 256-GRF mode);
A arrives as 32-row x 32-k 2D block reads, W as transposed 32-bit reads (16 rows x 8 words), which
is DPAS's B operand as loaded. No shared memory, no barriers. 8 subgroups per workgroup (128x128).

| m=512 GEMM | time (sustained, 200 runs) | TFLOPS |
| --- | --- | --- |
| 5120 -> 17408, workgroups ordered N-major | 1.583 ms | 57.6 |
| 5120 -> 17408, workgroups ordered M-major (the M tiles of one W column tile run back to back, W from L2) | **0.877 ms** | **104** |
| + 2D block prefetch two k-steps ahead | 1.086 ms | 84 (not kept) |
| 17408 -> 5120 (160 workgroups) | 1.248 ms | 73 |

For comparison, the same 5120 -> 17408 GEMM: infr Vulkan `_x256` ~2.0 ms sustained (Q4_K, its
in-shader decode included); llama.cpp Vulkan 1.55–1.60 ms in-stream. Short bursts run slower
(1.3 ms over 10 runs) while the clocks ramp.

So the native path is worth building: a ~50-line OpenCL C kernel is 2.3x our Vulkan GEMM. The
weights in a real model are quantized (Q4_K / IQ4_XS ~ 0.5–0.6 B per element against f16's 2), so
the decode path's bandwidth bound is lower, and dequantizing in registers before DPAS (or
TensorFold's int4-as-fp16-subnormal trick) is the next kernel question.

## Integration: hybrid (prefill GEMMs on Level Zero, the rest on Vulkan)

Chosen 2026-11-10 over a full fourth backend: Vulkan keeps every op, and the prefill GEMMs move to
Level Zero kernels that read the Vulkan buffers in place.

### Phase 1: interop (done, `crates/infr-vulkan/tests/ze_interop.rs`)

- **Memory**: a Vulkan buffer exported as a dma-buf (`VulkanBackend::p2p_export(.., DmaBuf)`) is
  imported with `Gpu::import_dmabuf` (`zeMemAllocDevice` + `ze_external_memory_import_fd_t`).
  Level Zero kernels read and write the Vulkan pages; 4 rounds of Vulkan-write → L0 read/write →
  Vulkan-read over 1M floats are exact, and the reverse direction too.
- **Ordering**: a Vulkan timeline semaphore (`tp_export_timeline`, `OPAQUE_FD`) is imported with
  `Gpu::import_vk_timeline` (`ZE_extension_external_semaphores`,
  `VK_TIMELINE_SEMAPHORE_FD`); `wait_semaphore` / `signal_semaphore` append GPU-side waits and
  signals to the immediate command list. Appending an unmet wait does not block the host.
- **fd ownership**: the Intel compute runtime (26.40) closes the fd it imports, for memory and
  for semaphores (measured: the descriptor is gone after a successful import). The infr-ze imports
  take an `OwnedFd`; callers pass a duplicate of the exporter's fd.

Per-handoff cost (B70 `03:00.0`, 200 rounds; one round is a 16 KB Vulkan copy, a 4096-element L0
add, a 16 KB Vulkan copy — two cross-API handoffs around ~µs of work):

| ordering | per round |
| --- | --- |
| Vulkan only: the same two submits chained by the timeline, no L0 | 74 µs |
| Vulkan → L0 → Vulkan, timeline semaphores (GPU-side) | 156–164 µs (63 µs of it host issue, overlapped) |
| Vulkan → L0 → Vulkan, host fences | 270–310 µs |

So an L0 segment between two Vulkan submits costs ~80 µs per round trip over a plain dependent
Vulkan resubmit — kernel-driver scheduling of a cross-queue dependency, not work. For 27B prefill
(64 layers), offloading every GEMM naively is ~8 handoffs per layer ≈ 512 per ubatch ≈ 20–30 ms
against ~690 ms per 512 tokens today; moving the norms/elementwise ops between GEMMs into L0 later
cuts that to ~2 per layer.

### Phase 2: quantized DPAS GEMMs (`kernels/gemm_q4k.cl` done)

The 27B (UD-Q4_K_M) GEMM work by weight format: IQ4_XS 36%, Q5_K 28%, Q4_K 25%, Q6_K 3.6%, the
rest under 3% each; the 4B Q4_K_M is nearly all Q4_K (+ Q6_K).

`gemm_q4k`: the f16 probe's tiling with the B operand dequantized in registers. Each lane owns two
W rows (DPAS's B operand is one column per lane) and turns 16 nibbles into 8 dwords of half
k-pairs: mask, widen bytes to words, `| 0x6400` (the half 1024 + q), `- 1024`, `* s + m`.

What it took (m=512, 5120 -> 17408, sustained):

| variant | time | TFLOPS | non-DPAS instr. per DPAS |
| --- | --- | --- | --- |
| OpenCL C `half16` dequant (`convert_half16(as_uchar16(..))`, `fma`) | 1.551 ms | 58.9 | ~21 |
| C^T = W·A^T (W as DPAS's A operand, whose layout is OpenCL's per-lane `half8`) | 1.812 ms | 50.4 | ~17 (the structure alone: 1.36 ms) |
| **vISA inline asm for the B operand** | **0.935 ms** | **97.6** | ~7 |
| bound: same kernel, no dequant (wrong values) | 0.800 ms | 114 | — |

OpenCL C keeps a per-lane `half16` de-interleaved in registers (element e of all 16 lanes in a
row), while DPAS's B operand wants each lane's two k-values packed in one dword; IGC moves every
value out and back. Inline vISA (`__asm__` with `"rw"` operands and `.decl ... alias`) computes the
packed rows directly: `mov (32) uw <- ub<4;2,1>` widens two nibble bytes into a dword's halves.
Two IGC inline-asm rules learnt: 32-wide ops in a SIMD16 kernel need `M1_NM` (the upper 16
channels are otherwise masked off), and there is no early-clobber constraint, so intermediates go
to asm-local `.decl` registers and only the last instruction writes the output.

Other shapes (m=512): 17408 -> 5120 1.070 ms (85 TFLOPS), 5120 -> 12288 0.577 ms (112),
5120 -> 6144 0.323 ms (100). m=2048 5120 -> 17408 is 4.85 ms (75): the 21 MB A no longer stays in
L2 across the N tiles — a packet-ordered grid (all N tiles of a 512-row M packet first) is the fix.
Against Vulkan `_x256` Q4_K (1.62 ms / ~2.19 ms for the two FFN shapes) this is 1.7-2x.

### Next

- The same for Q5_K, IQ4_XS (codebook: a byte-pair table instead of `| 0x6400`), Q6_K, Q8_0 and
  the long tail.
- Exportable allocations for the weights and the activation arena (dedicated, dma-buf), then the
  recorder split: end the command buffer at an offloaded GEMM, submit signalling the timeline, L0
  waits / runs / signals, the next command buffer waits.
