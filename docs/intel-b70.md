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
backend and level with its SYCL one** (149.2 vs 111.2 and 148.5 t/s), and
Qwen3.8-27B is ahead of Vulkan (28.8 vs 23.1; SYCL 29.9). A native Intel
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
| Qwen3.5-4B Q4_K_M pp512     | 1571           | **4178**  | 3165             | 7031           |
| Qwen3.5-4B Q4_K_M tg128     | 62.6           | **149.2** | 111.2            | 148.5          |
| Qwen3.8-27B UD-Q4_K_M pp512 | 220            | 739       | 746              | 1087           |
| Qwen3.8-27B UD-Q4_K_M tg128 | 13.8           | **28.8**  | 23.1             | 29.9           |

llama.cpp at `0f236bc21`; its 27B decode row is tg64. "infr first run" is
`f568ff7`. "infr now" is after plan slice 10. Prefill moved in slices 2, 9 and 10,
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

### Prefill: the 256-row XMX tile (plan slice 9)

The `_cm816` tile (128 rows x 256 columns, 1024 threads, BK=32) decoded each
weight block once per 128-row tile, with one thread per 32-element block: a
quarter of the threads did all the decoding while the rest waited at the
barrier. llama.cpp's Xe2 config (512 threads, 128 weight rows x 256 tokens,
32x32 warp tiles) reached ~53 TFLOPS on the same shapes against our ~19–23.

`_x256` (`gemm::native_gemm_warp_x256`, m >= 256): 256 rows x 128 columns,
BK=128, 512 threads as 32 warps of 32x32 (4x2 8x16 fragments), A converted to
f16 once and read from global (A_GLOBAL), so LDS holds only the 34 KB weight
tile. Every thread decodes exactly one block per stage, each block serves 256
rows, and there are a quarter of the barriers. Each output still accumulates
over K in 16-wide steps in order, so the generated text is byte-identical to
the old tile (checked on a 400-token prompt); `shared_decode_parity` gains an
m=300 case.

| m=512 GEMM (Qwen3.8-27B)    | `_cm816` | `_x256` | llama.cpp |
| --------------------------- | -------- | ------- | --------- |
| IQ4_XS 5120x17408           | 4.9 ms   | 2.3 ms  | 1.70 ms   |
| Q4_K 5120x17408             | 3.9 ms   | 2.2 ms  | 1.70 ms   |
| Q5_K 17408x5120             | 4.5 ms   | 2.7 ms  | 2.31 ms   |

| Qwen3.5-4B | pp128 | pp256 | pp512 | pp1024 | pp2048 |
| ---------- | ----- | ----- | ----- | ------ | ------ |
| before     | 1734  | 2411  | 2380  | 2319   | 1977   |
| `_x256`    | 1738  | 2952  | 3070  | 2874   | 2365   |
| llama.cpp  | 1394  | 3131  | 3165  | 3030   | 2802   |

Qwen3.8-27B pp512 376 → 558 (llama.cpp 746). What its pp512 spends now
(918 ms of GPU): the GEMMs ~65%, the 16 hd256 flash-attention layers 91 ms
(5.7 ms each), `deltanet_seq` 37 ms, the 5120x48 alpha/beta GEMVs 32 ms (the
per-row GEMV at m=512), IQ3_S's mmq GEMM 21 ms, and 396 `store_f16` A
conversions 18 ms.

### Prefill: XMX flash attention (plan slice 10)

Prefill attention on Xe2 ran `attn_nc_fa`, the scalar shared-memory flash kernel
written for devices without coopmat: the 16x16x16 flash kernels need a fragment
shape and a 32-lane subgroup Xe2 lacks, and qwen35's hd=256 is past their fixed
hd=128. At 512 tokens it took 3.8 ms per hd=256 layer on Qwen3.5-4B (llama.cpp:
1.2 ms) and 5.7 ms on Qwen3.8-27B.

`attn_xmx_fa.comp` is its drop-in on an 8x16x16 coopmat device (same push,
bindings and semantics: causal, sliding window, ring KV rows, GQA), selected
inside `Recorder::attention_prefill_nc_fa` for hd 128/256. 32 query rows x 32
keys per block, 256 threads (QK on 8 warps, PV on all 16); S and P go through
shared memory for the mask and the online softmax (8 lanes per row, clustered
reductions); O stays in registers for the whole kv loop, 4 fragments per warp.
A coopmat's element-to-row mapping is implementation-defined, so the kernel
reads it once from a probe matrix into a per-lane row/column table in shared
memory, used for the per-row rescale and the final bounded write. The first
version (128 threads, 8 O fragments per warp, the maps in registers) spilled
164 registers per Mesa's compile stats; this one 6. The recorder's
CPU-reference test (`attention_prefill_nc_fa_matches_cpu`, windows and ring
caches included) passes on it.

Also in this slice: consecutive `_x256` GEMMs on the same activation reuse its
f16 copy (`mmv_memo`; 396 → 253 `store_f16` per 27B pp512), the IQ4_XS/IQ4_NL
`_x256` decode reads the byte-pair codebook table from shared memory
(`-DIQ4_LUT`; IQ4_XS gate/up 2.6 → 2.3 ms, now level with Q4_K), and IQ3_S
gets an `_x256` build (it had only the int8 mmq GEMM: 7.2 ms per 17408x5120).

| prefill t/s    | before | 128 threads | 256 threads | llama.cpp Vulkan |
| -------------- | ------ | ----------- | ----------- | ---------------- |
| 4B pp512       | 3070   | 3504        | 3611        | 3165             |
| 4B pp2048      | 2365   | 3342        | 3609        | 2802             |
| 27B pp512      | 558    | 624         | 634         | 746              |

The `_x256` GEMMs still spill (~117 registers) — Q8_0, whose block decode is
trivial, spills the same, so it is the matmul loop: fully unrolled at BK=128
the compiler hoists every step's global A loads. A rolled k loop (4B pp512
2752), 2- or 4-step groups (Q4_K 5120x17408 1.95 → 5.3 / 4.4 ms) and a
1024-thread 32x16-warp tile (3370) were all slower. Measured alone, the m=512
5120x17408 GEMM runs 1.95–2.04 ms (45–47 TFLOPS) for Q4_K/Q5_K/Q8_0/IQ4_XS
against llama.cpp's 1.70.

**Split-K for small grids.** 512 threads and 35 KB of LDS fit two workgroups
per Xe core: 64 at once. The 5120-wide outputs at m=512 are 80 workgroups,
1.25 waves, so the tail idles most of the device (Q5_K 17408x5120 took 2.32 ms
against 1.96 for the same FLOPs at 5120x17408). `_x256_sk` splits K in two
when the grid is under two waves (`Recorder::matmul_native_x256_splitk`,
partial planes + the fixed-order `splitk_reduce`): 17408x5120 2.32 → 2.01 ms,
6144x5120 0.91 → 0.83; three or four splits lose it to the partial traffic.
4B pp512 3611 → 3686, pp2048 3609 → 3657; 27B pp512 634 → 645, pp2048 649.

**16x64 warp tiles.** The spills came from the A side: each 32x32 warp loaded
4 A fragments from global per k-step, and the fully unrolled loop hoisted all
32 of them. A 16-row x 64-column warp tile (16 x 2 warps over the same 256x128
tile, 2x4 fragments) halves the global A fragments per step and takes the
extra operand traffic from the LDS weight tile instead: spills 117 → 100, and
the m=512 5120x17408 GEMM Q4_K 1.95 → 1.62 ms (56.5 TFLOPS, past llama.cpp's
1.70), Q5_K 1.97 → 1.77, IQ4_XS 2.00 → 1.78; Q5_K 17408x5120 split in two
2.01 → 1.80 (llama.cpp 2.31). An 8x128 tile spills less (84) but is slower
(Q4_K 2.19 ms). Qwen3.5-4B pp512 3686 → 4026, pp2048 3657 → 3967;
Qwen3.8-27B pp512 645 → 707, pp2048 706.

**Narrow weights at prefill.** The 5120x48 DeltaNet alpha/beta (32-wide on the
4B) miss every GEMM tile (`gemm_ok` needs out_f % 64 == 0) and ran the per-row
GEMV: m x out_f workgroups each re-decoding the weight, ~340 µs each, 96 per
27B pp512. Two paths now, Xe2 only:

- m >= 256 (out_f % 16 == 0, under one 128-column tile): the `_x256` tile with
  a column guard (`_x256_cg`: columns past n stage zeros and are not stored),
  split 16 ways along K since its grid is only ceil(m/256) workgroups.
- 16 < m < 256: the multi-row f32 GEMV in 8-row blocks over one dispatch
  (`native_gemv_mrow.comp -DROWBLK`). A row-block-per-workgroup layout re-reads
  the activations per output, so at m=512 it only got 340 → 257 µs; the tile
  path above takes those shapes.

`shared_decode_parity` gains m=40 and m=300 cases at 512x48. Qwen3.5-4B pp512
4026 → 4138; Qwen3.8-27B pp512 707 → 735, pp2048 731.

**The f16 A copy written by its producer.** Every `_x256` GEMM input was a
separate `store_f16` pass (304 per 27B pp512, ~50 µs each). The prefill norm,
the DeltaNet gated norm, both SwiGLU forms and the attention sigmoid gate now
have `-DY16` builds that also write the f16 copy (the same `float16_t()` of the
same f32 value: a 400-token prompt's greedy output is byte-identical) and hand
it over through `mmv_memo` (`adapter::prefill_x256_linear_next`). 304 → 48
conversions. Qwen3.5-4B pp512 4138 → 4178, pp2048 4003 → 4130; Qwen3.8-27B
pp512 735 → 739, pp2048 731 → 741.

Where the GEMM tile stands: alone, the m=512 5120x17408 GEMM runs ~1.62–1.85
ms in short bursts and ~2.0 ms under sustained load (the card clocks down:
1.86 → 2.12 ms over 200 back-to-back runs); llama.cpp holds 1.55–1.60 ms
in-stream. With f16 weights (nothing to decode) the same tile is no faster
(1.79 ms), so the limit is the matmul pipeline (A from global, ~100 spills),
not the decode.

What is left on the 27B (846 ms of GPU per pp512): the GEMMs at ~40 TFLOPS
against llama.cpp's ~53 (5120x17408 2.3 vs 1.70 ms) are ~80% of it; then
`deltanet_seq` 37 ms, the 5120x48 alpha/beta projections 33 ms (they miss the
GEMM tier's out_f % 64 gate and run the per-row GEMV), attention 21 ms. A
vec4-wide weight-tile store in the GEMM was slower (601 → 589).

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

### Decode: the GEMVs were already at bandwidth; the gaps between them were not

Measuring a GEMV by host wall time over N recorded dispatches overstates it on
this backend: the recorder writes the whole command buffer before it submits,
about 20 µs of host time per dispatch, and that lands in the per-dispatch
figure. Decode replays a recorded tape, so it never pays it. Timing only the
submit and GPU wait (cold weights: six copies, 75+ MB, past L2) gives the real
mrow GEMV:

| GEMV (m=1)        | infr               | llama.cpp Vulkan |
| ----------------- | ------------------ | ---------------- |
| Q4_K 5120x17408   | 89.5 µs (560 GB/s) | 92.6 µs          |
| Q5_K 5120x17408   | 110 µs (555 GB/s)  | 118 µs           |
| IQ4_XS 5120x17408 | 98 µs (480 GB/s)   | 148 µs           |
| Q4_K 4096x14336   | 53.1 µs            | 54.4 µs          |
| Q4_K 14336x4096   | 58.9 µs            | 53.3 µs          |

(llama.cpp's figure is `test-backend-ops perf`, which reuses one buffer.) The
big shapes run at ~92% of the 608 GB/s spec. The earlier ~460 GB/s figures in
this file were host-inflated. Smaller shapes lose ~15 µs to the
last, partial wave: a 5120x5120 Q4_K GEMV is ~1.25 waves of 4-output
workgroups, and the tail wave is latency-bound (10 dependent load rounds per
lane). The 2x64 layout, SIMD16 and one-pass nibble loads did not move that
tail. A 2x-unrolled OUTS4 loop did (Q4_K 5120x5120 38.4 → 34.3 µs) but cost
Qwen3.5-4B 1.6% end to end (its in_f=2560 rows leave half an unrolled step
idle), so it was not kept; nor was the 256-thread skinny-GEMV build for the
5120x48 DeltaNet alpha/beta (~1 µs saved).

What decode did pay for was serialization: 453 barriers for ~558 dispatches
per Qwen3.5-4B token, ~10 of them small latency-floor kernels per layer. Plan
slice 7 removed some and cheapened others:

- **Activation quantization fused into its producer.** Every int8 GEMV input
  was a separate `quant_q8` dispatch between the producing op and the GEMV.
  Skipping it outright (wrong output, same work otherwise) measured the
  ceiling: 4B 142.1 → 148.3, Qwen3-8B 92.1 → 95.6. The decode norm, both
  SwiGLU forms, the DeltaNet gated norm and the attention sigmoid gate now have
  `_q8` builds that also write `qa`/`dact`/`sact` (quant_q8's math, one wave32
  per 32-element block) and hand them over through `mmv_memo`
  (`adapter::decode_int8_linear_next`); the fused-residual GEMV honors the
  memo too. Xe2 only (`caps::decode_quant_fusion`): the fused norm sums its
  squares in a different order, so other devices keep their bits until
  measured. The decode norm itself is `rmsnorm_q8.comp`: dim/128 workgroups
  that each reduce the L2-resident row instead of the single latency-bound
  `rmsnorm_wide` workgroup (a single-workgroup norm+quant was slower than the
  two dispatches). What is left is one `quant_q8` per layer on dense models,
  the attention output into `o_proj`. `tests/rmsnorm_q8_parity.rs` checks the
  int8 output is bit-exactly `quant_q8` of the norm's own f32 row.
- **Clustered OUTS4 reduction on Xe2.** `subgroupClusteredAdd` over each
  output's 16 lanes instead of the shared tree and its barriers: IQ4_XS
  5120x17408 105 → 98 µs, Q8_0 173 → 167, Q6_K 137 → 132 (`caps::mrow_cluster`;
  every OUTS4 build has a `_cl` twin, so decode and verify stay on one kernel).
- **Q3_K and Q8_0 on the int8 decode set on Xe2**, above 8M elements. The f32
  GEMV is the slow side here (Q8_0 5120x17408 238 vs 173 µs; Q3_K ~165 vs
  ~540 GB/s). The floor keeps small models such as the 0.6B Q2_K mix on f32.

tg128 on `Vulkan1`, each step on top of the previous:

| step                                    | Qwen3.5-4B | Qwen3-8B | Qwen3.8-27B |
| --------------------------------------- | ---------- | -------- | ----------- |
| slice 6                                 | 142.1      | 92.1     | 27.2        |
| Q3_K/Q8_0 int8, norm/SwiGLU/gate `_q8`  | 146.6      | 93.9     | 28.1        |
| clustered OUTS4 reduction               | 149.2      | 94.7     | 28.4        |

(Only the 27B carries Q3_K/Q8_0 weights large enough for the int8 floor.)

### Decode: where the 27B's time goes now (plan slice 8)

Running only the `Linear` ops (every other op skipped; a measurement build, not
shipped) puts Qwen3.8-27B's GEMV stream at 31.7 ms of the 35.2 ms token, and
the other ops alone at 2.1 ms. Skipping one GEMV shape at a time attributes the
31.7 ms; the ideal is the weight bytes at 560 GB/s:

| shape (per token)               | in-stream | ideal   | efficiency |
| ------------------------------- | --------- | ------- | ---------- |
| ffn gate/up 5120x17408 (128)    | 13.0 ms   | 11.7 ms | 90%        |
| ffn down 17408x5120 (64)        | 7.3 ms    | 6.0 ms  | 83%        |
| out projections 6144x5120 (64)  | 2.7 ms    | 2.5 ms  | 92%        |
| qkv, z, q, k/v                  | ~ideal (they overlap their siblings) | | |
| lm_head                         | 1.8 ms    |         | 575 GB/s   |

What the losses are:

- **IQ3_S had no int8 path.** Its 4 FFN weights ran the f32 GEMV at ~180 GB/s.
  It now has an int8 mrow build (`FMT_IQ3S`: the 512-entry grid in shared
  memory, a byte-wise sign negate, one scale per sub-block), on the Xe2 decode
  set: 5120x17408 213 → 78 µs, 17408x5120 205 → 87. Its m=3..8 rows follow
  (a decode-set dtype is int8 in prefill too, `adapter::mrow_int8_dtype_ok`),
  and the MTP verify batch takes the same kernel as decode
  (`tests/mmv_row1_bit_identical.rs`, `tests/mmv_mrow_iq3s.rs`). 27B tg128
  28.4 → 28.8; MTP 40.6 t/s at α 0.80.
- **IQ4_XS's codebook lookup** (61 of the 128 gate/up weights): the kernel runs
  at ~485 GB/s and at 560 with the lookup removed. The cost is the 16
  shared-memory reads per 32 elements, not bank conflicts (conflict-free
  addresses: no change), latency (2x unroll: no change), or the header loads
  (two-word header: no change). An ALU select chain (123 µs) and a half-ALU
  hybrid (105 µs) are slower than the table (98 µs). ~0.8 ms per token, open.
- **Q5_K at the down shape** (22 weights) runs at ~512 GB/s on the 2x64 layout;
  OUTS4 there is slower for every format (Q5_K 123 µs, IQ4_XS 113 vs 95).
- The DeltaNet alpha/beta skinny GEMVs (96 per token) overlap their siblings:
  skipping them outright is worth only 0.7% (4B 1.1%).
- A 2-output x 32-lane layout made the bench shapes faster (Q4_K 5120x17408
  90.3 → 85.7 µs, 585 GB/s) but decode slower (4B 149.2 → 146.2, Qwen3-8B
  94.7 → 89.8; the 27B unchanged) — not kept.

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
6. **Done.** Plain decode, three changes (measured on `Vulkan1`, tg128):
   the m=1 decode reuses the previous Linear's quantized activation
   (`mmv_memo`; 381 → 251 `quant_q8` per 27B token), the strided DeltaNet
   form is the decode default (no `CopyStrided` unpack; 144 dispatches per
   27B token), and the OUTS4 mrow layout covers in_f <= 5120 and Q6_K on Xe2
   (`caps::mrow_outs4`). Qwen3.5-4B 120.0 → 141.8 t/s, Qwen3.8-27B 25.3 → 27.1.
   The spec bandwidth (608 GB/s) puts the single-card ceilings at ~243 and
   ~40 t/s; the cold-weight mrow GEMV still runs 220–475 GB/s.
7. **Done.** Decode dispatch count, above: `quant_q8` fused into its
   producers, the clustered OUTS4 reduction, Q3_K/Q8_0 int8 on Xe2.
   Qwen3.5-4B 142.1 → 149.2 t/s, Qwen3-8B 92.1 → 94.7, Qwen3.8-27B 27.2 → 28.4.
   The load-shape idea this item listed (Q4_K lanes reload the qs pair) was
   measured and is not the limit; the mrow GEMV is at ~560 GB/s on big shapes.
8. **Done.** The IQ3_S int8 mrow build, and the 27B decode attribution above.
   Qwen3.8-27B tg128 28.4 → 28.8.
9. **Done.** Prefill: the `_x256` tile, above. Qwen3.5-4B pp512 2380 → 3070,
   Qwen3.8-27B 376 → 558.
10. **Done.** XMX flash attention, the f16 conversion memo, the IQ4 table in
    the GEMM decode, IQ3_S `_x256` (above). Qwen3.5-4B pp512 3070 → 3611,
    pp2048 2365 → 3609; Qwen3.8-27B pp512 558 → 634.
11. **Done.** Split-K for the `_x256` GEMMs whose grid is under two waves.
    Qwen3.5-4B pp512 3611 → 3686; Qwen3.8-27B 634 → 645.
12. **Done.** 16x64 warp tiles for `_x256`, and the narrow prefill weights
    (column-guarded `_x256_cg`, row-blocked multi-row GEMV): Qwen3.5-4B pp512
    3686 → 4138, Qwen3.8-27B 645 → 735.
13. **Done.** The producers write the `_x256` GEMMs' f16 A copy (`-DY16`):
    Qwen3.5-4B pp512 4138 → 4178, pp2048 → 4130; Qwen3.8-27B 735 → 739.
14. Next: the `_x256` matmul pipeline (A through LDS with a partial-block
    decode, as llama.cpp's mul_mm stages both operands), then the IQ4_XS
    decode lookup and the attention output → `o_proj` quant on dense models.

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

| engine           | plain | MTP n_max=3     | MTP n_max=6 |
| ---------------- | ----- | --------------- | ----------- |
| infr Vulkan      | 25.3  | 41.4 (adaptive) | 38.0        |
| llama.cpp Vulkan | 23.0  | 44.2            | 41.9        |
| llama.cpp SYCL   | 29.5  | 72              | **80**      |

Same acceptance (llama.cpp 159/233 = 0.68), so the gap is cycle cost. An m=7
infr forward costs 1.73x an m=1 forward (`pp7` 72.8 ms vs `pp1` 42 ms) because
the int8 mrow GEMV's dp4a work scales with m on Xe2. Each draft step also
re-reads the 1 GB vocab head (2.2 ms), and a cycle spends ~20 ms on host round
trips: a re-recorded static verify graph and a staged hidden-row download.
