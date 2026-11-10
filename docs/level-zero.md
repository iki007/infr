# Level Zero backend (`infr-ze`) — Intel Xe2, native

Status: **phase 0 done** (2026-11-10). The device layer and a DPAS GEMM probe run on the Arc Pro
B70; nothing in the engine uses the crate yet. Backlog B102.

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

## Next

The integration choice is open (see the session notes / the user's call):

- **Prefill GEMMs on Level Zero, everything else on Vulkan**: the Vulkan arena and activations
  exported as dma-buf and imported into Level Zero on the same device; per-format dequant-in-kernel
  DPAS GEMMs; Vulkan timeline semaphores and Level Zero events for ordering.
- **A full fourth backend**: every op of the graph in OpenCL C, model family by model family.
