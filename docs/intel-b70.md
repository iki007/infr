# Intel Arc Pro B70 (Xe2) — first run, where the time goes, the plan

Campaign log for infr on Intel Battlemage. First measured 2026-11-09 at
`f568ff7`. The dev box now has **two Arc Pro B70** cards (Battlemage G31, PCI id
`8086:e223`, 32 GB each, Mesa 26.2.4 ANV). Earlier docs describe an RX 7900 XTX
here; for Intel behaviour, this file supersedes them.

## Status

infr builds and runs on Xe2 and generates coherent text on both cards (Qwen3.5-4B
and Qwen3.8-27B checked by hand). The first run was **2–3.4x behind llama.cpp's
Vulkan backend on prefill and ~1.7x behind on decode**. The causes are measured
(below), and every one of them is inside the existing Vulkan backend. Plan
slices 1–3 have landed. The GPU suite is clean. Prefill runs on XMX through an
8x16x16 coopmat tile (+53% / +71% pp512), still 1.3x / 2x behind llama.cpp's
Vulkan prefill. Decode on Qwen3.5-4B is now **ahead of llama.cpp's Vulkan
backend** (120 vs 111 t/s), and so is Qwen3.8-27B (25.3 vs 23.1 t/s). A native Intel
backend (Level Zero) is deferred until the Vulkan work is done (`backlog.md`
B102).

## Running on this box

| handle              | PCI     | role                                              |
| ------------------- | ------- | ------------------------------------------------- |
| `Vulkan0`           | 0a:00.0 | drives the display; behind the PCH (~7 GB/s PCIe) |
| `Vulkan1`           | 03:00.0 | headless, full link — **measure here**            |
| SYCL `level_zero:0` | 03:00.0 | the same card as `Vulkan1`, for llama.cpp SYCL    |

infr's default (no `--dev`) is the first discrete device, which is the display
card (`backlog.md` B104). Pass `--dev Vulkan1` / `INFR_DEV=Vulkan1`. Strata's
measurements on this box found ~2x the per-kernel launch gap on the display
card. The two cards have no P2P path to each other.

The GPU parity tests are `#[ignore]`d; the full run is

```bash
INFR_DEV=Vulkan1 cargo test --release -p infr-vulkan --no-fail-fast -- --include-ignored --test-threads=1
```

## Numbers (2026-11-09, `Vulkan1`, `-fa 1` for llama.cpp)

| model                       | infr first run | infr now  | llama.cpp Vulkan | llama.cpp SYCL |
| --------------------------- | -------------- | --------- | ---------------- | -------------- |
| Qwen3.5-4B Q4_K_M pp512     | 1571           | 2380      | 3165             | 7031           |
| Qwen3.5-4B Q4_K_M tg128     | 62.6           | **120.0** | 111.2            | 148.5          |
| Qwen3.8-27B UD-Q4_K_M pp512 | 220            | 376       | 746              | 1087           |
| Qwen3.8-27B UD-Q4_K_M tg128 | 13.8           | **25.3**  | 23.1             | 29.9           |

llama.cpp at `0f236bc21`; its 27B decode row is tg64. "infr first run" is
`f568ff7`. "infr now" is after plan slice 3. Prefill moved in slice 2,
`INFR_NO_COOPMAT=1` still gives the first-run prefill numbers, and that is how
the A/B was taken. infr's 27B prefill
arms the submit splitter (`split/185`, `split/244`). That is a symptom of the
slow forward, not a cause: the splitter's cap is measured from forward time.

## Test suite (2026-11-09)

The first run: 284 pass, 17 fail. None of the 17 was a production-path failure,
and all 17 are fixed (plan slice 1); the suite now passes on the B70. What they
were:

- **12 coopmat tests run a 16x16x16 kernel the device does not enumerate**
  (`coopmat_gemm_tiled_matches_cpu`, `warp_gemm_matches_cpu`, the
  `attention_prefill_*` and `attn_flash_*` parities, `matmul_proj`,
  `matmul_native_splitk_f16`, `native_gemm_warp_iq4nl`). They call the kernels
  directly without checking `caps`. ANV does not reject the pipeline; it
  computes wrong numbers (rel. error 0.1–0.5). Production never dispatches
  them here because shape selection finds no match.
- **4 `mmv_mw` tests ask for subgroup-16 builds that do not exist**
  (`native_mmv_mw_spv(.., sg16 = true)` returns `None` for most dtypes, which
  panics in `Recorder::linear_mmv_mw_at`). Xe2 is the first device here where
  `sg16()` is true. Decode no longer routes through `mmv_mw`, so this is
  test-only.
- **`ep_matches_single_device_synthetic` compares device NAMES** to prove it
  got two distinct GPUs; two identical cards share a name. The two-device math
  itself matched (max |Δ| 2.4e-7).

## Where the time goes

### Prefill: cooperative matrix was dark

ANV on Xe2 enumerates exactly these coopmat configs (`INFR_DEBUG_COOPMAT=1`):

```text
(8,16,16) f16×f16→f16, f16×f16→f32
(8,16,16) bf16×bf16→bf16, bf16×bf16→f32
(8,16,32) s8×s8→s32, u8×u8→u32
```

infr's coopmat kernels exist only at 16x16x16 (every production shader) and
8x8x16 (the opt-in `_cm8` builds for Alchemist). Neither matches, so
`select_coopmat_shape` returns `None` for every tier and the banner reads
`f16cm:n bf16cm:n i8cm:n`. The `INFR_CM_8X8=1` opt-in does not help: Xe2
enumerates no 8x8x16 config. `caps.rs`' trust rule already classifies the part
correctly (`IntelXe2`, `Enumerated`). The gap is the shape, not the trust.

Qwen3.5-4B pp512, `INFR_PROF_OPS=1`: 81% of device time is the dp4a MMQ tier
(`native_gemm_mmq_q4k_streamed` 55%, `_q6k_streamed` 14%, `_q5k` 12%), which
reaches ~16 TOPS. llama.cpp builds its coopmat shaders with the tile as
specialization constants (`TM/TN/TK` from the enumeration), so it runs XMX at
8x16x16 here.

**Landed (plan slice 2):** `select_coopmat_shape` takes 8x16x16 for f16
(`COOPMAT_TILE_8X16`, `caps.f16_coopmat_8x16()`), and `native_gemm_warp` has
`_cm816` builds for Q4_K, Q5_K, Q6_K, Q8_0, Q2_K, Q3_K, IQ4_XS and IQ4_NL. They
ride the adapter's existing `cm8_ok` XMX tier, and `gemm::native_gemm_warp_xmx`
owns their tiles. Other formats stay on `nc_mmq`. f16 activations make these
GEMMs closer to the host oracle than the int8 tier: ~3e-4 relative against
~3.5e-3, from `vulkan_linear_all_weight_quants_gemm_tier_match_the_host_oracle`.
Greedy output on Qwen3.5-4B is identical with and without the tier. The banner
now prints `f16cm:8x16`.

The tile is the part that mattered. Reusing `_cm8`'s geometry (BM=32 x BN=128,
8 warps of 16 lanes) made prefill **slower** than dp4a. Larger tiles at 8 warps
were slower still: more fragments per warp spill Xe2's registers. What works is
llama.cpp's shape: many small warps. Q4_K, m=512, Qwen3.5-4B, ms per GEMM:

| tile                                        | 2560x18432 | 9216x2560 |
| ------------------------------------------- | ---------- | --------- |
| `nc_mmq` (dp4a, the previous default)       | 3.1        | 1.6       |
| 32x128, 8 warps x 16 lanes (`_cm8` shape)   | 4.3        | 2.8       |
| 64x256, 8 warps x 32 lanes                  | 7.5        | 3.6       |
| 128x256, 8 warps x 32 lanes                 | 20.1       | 7.7       |
| 128x128, 32 warps x 16 lanes (512 threads)  | 2.1        | 1.6       |
| 128x256, 64 warps x 16 lanes (1024 threads) | **1.8**    | **1.4**   |

The last row is `_cm816`, and `_cm816_n128` is the 512-thread row for n%256 != 0.
Both use the shader's new `XE_WM`/`XE_WN` per-warp overrides and a `ROW_GUARD`,
because C is padded to 64 rows, not 128. Still open: the remaining 1.3x / 2x to
llama.cpp. Candidates are an f16-A (`A_GLOBAL`) variant to drop the As stage, the
attention and DeltaNet prefill kernels, and the formats without a build.

### Decode: three ops were the whole gap

Qwen3.5-4B, one decoded token. infr from `INFR_SEAM_NO_REPLAY=1
INFR_PROF_OPS=1 INFR_PROF_OP_SHAPES=1`; llama.cpp from `GGML_VK_PERF_LOGGER=1`.
Same card.

| op                                 | llama.cpp          | infr               | ratio  |
| ---------------------------------- | ------------------ | ------------------ | ------ |
| lm_head Q6_K 2560x248320 (tied)    | 1129 µs (461 GB/s) | 6681 µs (78 GB/s)  | 5.9x   |
| gated DeltaNet, 24 layers          | 142 µs             | 1901 µs            | 13x    |
| attn_qkv Q5_K 2560x8192, per layer | 32 µs              | 59 µs              | 1.8x   |
| ffn gate+up Q4_K / ffn_down Q6_K   | 2x34 / 58 µs       | 67 (fused) / 43 µs | parity |

The first three sum to ~7.9 ms of a ~15.9 ms token, more than the whole gap to
llama.cpp's 10.1 ms.

- **lm_head — not the kernel, its output.** Isolated, the same GEMV at the same
  shape runs in 1152 µs (452 GB/s, llama.cpp's speed) into a device buffer, and
  in **7031 µs into a `Readback` buffer**. The seam bound the decode logits to
  its host-visible `logits_buf` even when the step samples on the GPU and never
  downloads them. On this card the LM head's 248k scattered stores into system
  memory were the whole 5.5 ms. That also explains the 174 µs `argmax_part`:
  it read the row back over PCIe. **Fixed (slice 3):** the session has a
  device-local `logits_dev`, bound whenever the call samples on the GPU and
  nothing wants the row. The host-sampled path still writes `logits_buf`
  (B106). Ruled out on the way: OUTS4 `o4` for tall shapes (worse, 51.7), the
  2-D `dispatch_wide` split (no change), 16/32-thread mrow workgroups (no
  change).
- **DeltaNet — fixed (slice 3).** The old `deltanet.comp` gave one thread a
  whole state column and looped over `kd` serially in 32-thread workgroups.
  llama.cpp spreads each column's reduction over 16 lanes, but its state is
  k-contiguous and infr's is d-contiguous. The rewrite puts 16 adjacent columns
  on the lanes (coalesced) and 8 row groups per workgroup. The state stays in
  registers for the whole `rows` loop, and the 8 group partials combine in
  fixed order. Result: 79 → 12.6 µs per layer; the same kernel for decode and
  verify.
- **attn_qkv — fixed (slice 3).** Q5_K (and, on the 27B, IQ4_XS) rode the f32
  GEMV, which runs at ~176 GB/s here against the int8 mrow kernel's ~410. The
  int8 decode set is AMD-measured (`adapter::mmv_int8_decode_dtypes`), and
  there Q5_K int8 was a −1.4% wash. On `DeviceArch::IntelXe2` the set now also
  has Q5_K and IQ4_XS (IQ4_XS gained the `_res` mrow twins it lacked). 27B decode
  15.7 → 18.8, 4B 115.7 → 120.0.

- **IQ4_XS / IQ4_NL codebook — fixed (slice 4).** The mrow decode did two
  dynamic codebook lookups per `qs` byte. A 256-entry byte-pair table in shared
  memory (`KVP`, one load per byte; bit-identical) took the cold-weight GEMV,
  in GB/s: IQ4_XS 5120x17408 230 → 290, 17408x5120 187 → 316; IQ4_NL 256 → 350
  and 209 → 307. 27B decode 18.8 → 25.3.

Where decode stands against the hardware: Qwen3.8-27B reads ~15.3 GB of weights
per token, so 25.3 t/s is ~390 GB/s effective. The practical streaming ceiling
on this card is ~560 GB/s (Strata), about 36 t/s single-stream.

Smaller items: the activation quantizer is 144 dispatches per token at ~7 µs
each (B105). The `argmax_part` row (174 µs) is probably the profiler charging
the lm_head tail to the next stamp; it is a 1 MB scan, so confirm it before
working on it.

### Detection

- `cores:?`: the B70 (`0xE223`, 32 Xe cores) and B65 (`0xE222`, 20) are missing
  from `intel_shader_core_count`. llama.cpp's table has both.
- `VK_NV_cooperative_matrix2` is advertised without
  `cooperativeMatrixWorkgroupScope`; `check_coopmat2_support` correctly refuses
  it.
- `maxComputeSharedMemorySize` is 48 KB per workgroup here (64 KB on RDNA3).

## What the other engines on this box show

**llama.cpp Vulkan** (`ggml-vulkan.cpp`, Intel arms):

- coopmat tile from the enumeration; dedicated Intel warptiles (`l_warptile =
  {512,128,128,16,..}`, mmq `{512,128,256,32,..}` on Xe2). Each pipeline's
  required subgroup size is pinned to its warptile's WARP element.
- For quantized weights on Intel coopmat1, it prefers f16 activations over the
  q8_1 integer path (`prefer_f16_b`).
- Mat-vec: subgroup 16 on Intel, 2 rows per workgroup for the legacy quants.
  On Xe2 it always uses MMVQ (int8 dot) for Q2_0/Q2_K/Q3_K/Q6_K; for other
  types only from k >= 2048.
- A separate two-phase flash-attention decode pipeline for Xe2. Subgroup
  collectives are off on Intel (PR 14316).

**llama.cpp SYCL**: reorders weights to struct-of-arrays for its MMVQ ("Reorder:
Y") and runs prefill GEMMs through oneMKL/oneDNN on XMX. That is the 1.5–2.2x
prefill lead over its own Vulkan backend.

**Strata's SYCL port** (`docs/INTEL.md` in that repo, measured on these cards):

- Practical streaming ceiling ~560–580 GB/s. Its native mat-vecs reach
  290–323 GB/s at one column and fall with more columns: decode is ALU-bound.
- XMX (DPAS) mat-vec for IQ4_XS gave 1.06–1.37x, only via a local-memory decode
  table.
- Arc's fp32 divide is not correctly rounded by default (OpenCL allows 2.5
  ulp). Its byte-exact `amax / 127` quantizer had 303k mismatches until it
  built with `-cl-fp32-correctly-rounded-divide-sqrt` (B103).
- A device spin that never sees its flag gets the GT reset by the `xe` driver,
  and the card stays wedged until reboot. Bound every spin.
- `SYCL_CACHE_PERSISTENT=1` segfaults on Xe2.

**TensorFold's XPU port** (`docs/xpu/` in that repo): drives the B70 through
**Level Zero via `dlopen`**, with no SYCL and no C++ at run time. Kernels are
OpenCL C compiled by `ocloc`, and the prompt GEMMs are generated as ISA by
nGEN.

- The gate|up GEMM sustains 146 TFLOPS fp16 and ~209 TOPS int8.
- Prefill is power-bound at 230 W (~2180 MHz, not the 2600 of pure DPAS).
- Everything runs SIMD16 with DPAS and 2D block loads.
- int4 weights enter DPAS as fp16 subnormals (scale × 2^24), so the W4 GEMM
  needs no dequant pass.
- The residual add, RMSNorm and activation quantization are one kernel.

This is the shape a pure-Rust native Intel backend would take (B102).

## Plan

Each slice is measured against the numbers above on `Vulkan1`.

1. **Done.** Detection and tests: the B70/B65 core counts and the 17 test-side
   failures. Coopmat tests skip without the 16x16x16 shape, the Q5_K sg16
   `mmv_mw` arms are wired, two host-reference metrics now tolerate
   cancellation zeros, and the two-device test no longer compares names.
2. **Done.** Prefill: the 8x16x16 `_cm816` XMX tile (above). +53% pp512 on
   Qwen3.5-4B, +71% on Qwen3.8-27B.
3. **Done.** Decode: the DeltaNet rewrite, device-local decode logits, and the
   Xe2 int8 decode set. Qwen3.5-4B tg128 62.6 → 120.0, past llama.cpp's 111.2;
   Qwen3.8-27B 13.8 → 18.8.
4. **Done.** The IQ4 codebook byte-pair LUT: Qwen3.8-27B tg128 18.8 → 25.3.
5. **Done (opt-in).** MTP speculative decode, below. Qwen3.8-27B 25.3 → 38.0
   t/s under `INFR_MTP=1`.
6. Next: the MTP cycle cost (an XMX small-m verify GEMV, the per-cycle host
   round trips, the draft head), then prefill.

## MTP (speculative decode, `INFR_MTP=1`)

Re-enabled as an opt-in after four fixes, each measured on Qwen3.8-27B
UD-Q4_K_M (its GGUF ships the `nextn` head), greedy, 200 tokens:

| step                                                                                                                   | t/s                                |
| ---------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| the parked path, kill switch flipped                                                                                   | OOM (trunk weights uploaded twice) |
| warmup no longer builds the plain session                                                                              | 17.1                               |
| `MROW_BANDS_XMX`: verify m=5..16 at wide n on the mrow GEMV, not the 128-row `_cm816` tile (1.3 ms → ~0.2 ms per GEMM) | 32.2                               |
| device-local verify logits on the GPU accept path                                                                      | 33.3                               |
| persistent trunk session across turns (was a ~15 GB re-upload per call; TTFT 12.4 s → 0.4 s)                           | 38.0                               |

Verify now also runs decode's own `deltanet` kernel (m <= 16), and takes the
int8 mrow exactly where decode does (`mmv_mw_choice`), which lifted α from 0.66
to 0.71 at 6 drafts. The draft length is adaptive with a cap of 3 (the measured
best; llama.cpp's default too):

| Qwen3.8-27B greedy, 200 tok | plain | MTP cap 6 fixed | MTP cap 3 fixed | MTP cap 3 adaptive |
| --------------------------- | ----- | --------------- | --------------- | ------------------ |
| code prompt                 | 25.3  | 39.0            | 42.4            | 41.4               |
| prose prompt                | 25.3  | 15.7            | 22.9            | 24.5               |

On prose MTP still only breaks even. After a partial accept, DeltaNet's state
cannot be rewound, so the next verify re-prefills from the last clean snapshot
(m=13 at ~100 ms). llama.cpp writes a state snapshot per verified row instead
(its `K` slots in `gated_delta_net.comp`), which keeps every verify at drafts+1
rows. That is the next MTP change, with an XMX small-m verify GEMV. Greedy MTP
output can still differ from plain on a near-tie; on the code prompt it does at
cap 3 (one identifier, ~20 tokens in).

Against llama.cpp `0f236bc21` with `--spec-type draft-mtp` on the same card
(llama-server, the same prompt, greedy):

| engine           | plain | MTP n_max=3 | MTP n_max=6 |
| ---------------- | ----- | ----------- | ----------- |
| infr Vulkan      | 25.3  | —           | 38.0        |
| llama.cpp Vulkan | 23.0  | 44.2        | 41.9        |
| llama.cpp SYCL   | 29.5  | 72          | **80**      |

Same acceptance (llama.cpp 159/233 = 0.68), so the gap is cycle cost. An m=7
infr forward costs 1.73x an m=1 forward (`pp7` 72.8 ms vs `pp1` 42 ms) because
the int8 mrow GEMV's dp4a work scales with m on Xe2. Each draft step also
re-reads the 1 GB vocab head (2.2 ms), and a cycle spends ~20 ms on host round
trips: a re-recorded static verify graph and a staged hidden-row download.
