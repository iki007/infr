# DFlash2 speculative decoding

DFlash2 drafts a whole block of tokens in one forward of a small block-diffusion model. The target
then verifies the block in one batched forward. infr runs it on the Vulkan seam:

```sh
INFR_DFLASH=/path/Qwen3.8-27B-DFlash2-body-Q4_K_M-selfhead-head98k-Q4_0.gguf \
  infr run /path/Qwen3.8-27B-UD-Q4_K_M.gguf
```

`INFR_DFLASH` (config `spec.dflash`) names the drafter GGUF (`general.architecture = dflash`). The
loader checks it against the target: the target layers it taps must exist, and its width, vocabulary
and selector tables must match. Greedy (`--temp 0`) commits exactly the target's argmax stream.
`temp > 0` uses the speculative-sampling accept rule, so the output is a sample from the target's
own distribution. The greedy-text caveat MTP documents applies here too: the verify batch and
single-row decode round differently, so a near-tie can flip a token.

Code: `crates/infr-llama/src/dflash.rs` (drafter, graphs, selector, the spec loop). Reference
semantics: llama.cpp's `src/models/dflash.cpp` (`graph<false>`, `build_dflash2_conv`,
`build_dflash2_selector`) and `common/speculative.cpp` (`common_speculative_impl_draft_dflash`).

## How a cycle runs

1. **Draft.** The block `[anchor, mask × 7]` sits at positions `pos0 .. pos0+8`, where the anchor
   is the last committed token and the target has not processed it yet. It goes through the
   drafter's 5 layers:
   - Each attention and FFN half wraps the dynamic two-tap convolution
     (`Op::BlockConv`, group 16).
   - The block attends bidirectionally to itself and to the injected context
     `[pos0+1-window, pos0)`. This is `AttnMask::Canvas`: one shared lower bound, with the
     K/V cache a ring over the 2048-token window.
   - The 98,580-row head plus the rank-256 selector gate come back to the host.
   - The host takes the top-16 per position, maps them to target ids (`d2t`), and scores
     `successor[cand] · (predecessor[prev] ⊙ gate)` + logit.
   - A greedy walk (or a sampled one at `temp > 0`) reads the draft off the lattice.
2. **Verify.** One target VERIFY forward runs over `[anchor, drafts]`, 8 rows, with a per-row
   argmax on the GPU (full logits at `temp > 0`, written device-local and copied back once).
   Embeddings are gathered on the GPU.
3. **Inject.** Every target forward copies the inputs of the tap layers (`dflash.target_layers`:
   6, 20, 34, 48, 62 on the 27B) into a position-indexed ring on the target session
   (`SeamKv::enable_taps`, `[2120 slots, 5 × 5120]` f32). After the verify, the accepted rows go
   through `fc` + `enc.output_norm` and each drafter layer's `wk`/`k_norm`/RoPE and `wv`, into the
   drafter's K/V at their positions. The prompt prime injects its last window.
4. **Roll back the recurrence.** Qwen3.8's DeltaNet layers cannot drop rejected rows by cache
   truncation. The VERIFY forward records each layer's state after every row (`Op::DeltaNet::hist`,
   the `deltanet_hist` kernel) plus its conv inputs (`SeamKv::spec_hist`). After the accept,
   `spec_hist_select` copies the accepted row's state back in one batched submission
   (`Backend::copy_buffers`). Every verify is therefore exactly `1 + drafts` rows, with no
   snapshot and no re-prefill. MTP's snapshot/restore/re-prime remains the fallback.

## Numbers

Arc Pro B70 (`Vulkan1`), Qwen3.8-27B UD-Q4_K_M, the shipped DFlash2 Q4_K_M drafter, 256 tokens on
the reasoning-heavy prompt in `crates/infr-llama/tests/dflash.rs`:

| engine | mode | t/s | accept rate (drafted tokens) |
|---|---|---:|---:|
| infr | plain decode | 28.7 | — |
| infr | DFlash, greedy | 39.9 (1.39×) | 0.36 |
| infr | DFlash, temp 0.6 / top-k 20 / top-p 0.95 | 41.2 (1.44× plain sampled) | 0.40 |
| llama.cpp Vulkan | DFlash, greedy, same rendered prompt | 25.4 | 0.30 |

That prompt drafts poorly on both engines; code continues further per block.

Each greedy cycle costs about 85 ms:

| step | time |
|---|---:|
| 8-row verify | ~70 ms |
| drafter block forward | ~9 ms |
| host selector | ~3 ms |
| injection | ~2 ms |
| state select | ~1 ms |

The verify and the drafter are both bound by the 8-row multi-row GEMVs. They run at about
250–430 GB/s, against about 470 GB/s for single-row decode and 608 GB/s peak. That kernel is the
next lever, and it would speed up MTP as well.

The path there:

| change | t/s |
|---|---:|
| first end-to-end run | 21.6 |
| + GPU embed gather in VERIFY (the host table took 2.3 s and 5 GB on first use) | 29.5 |
| + batched snapshot copies, then the per-row state history (no re-prime, verify always 8 rows) | 39.9 |

## Testing

- `crates/infr-llama/tests/seam_op_parity.rs`:
  - `block_conv_parity`: CPU and Vulkan against a transcription of `build_dflash2_conv`.
  - `deltanet_hist_parity`: CPU vs Vulkan history, and recording it leaves the output unchanged.
- `crates/infr-llama/tests/dflash.rs` (`--ignored`; needs `INFR_DFLASH_TARGET`, `INFR_DFLASH_DRAFT`
  and a GPU) runs plain greedy and DFlash greedy and prints speed and acceptance:
  - `INFR_DFLASH_TEMP=0.6` measures the sampled path.
  - `INFR_DFLASH_SKIP_PLAIN=1` skips the reference run.
  - `INFR_DFLASH_DUMP_PROMPT=<file>` writes the rendered prompt, for the llama.cpp comparison:
    `llama-server -md <drafter> --spec-type draft-dflash --spec-draft-n-max 7`, then
    `/completion` with the same text.

## Limits and next steps

- **Platform:** Vulkan only. Metal refuses `Op::BlockConv` and the DeltaNet history. The CPU
  backend implements both ops, but there is no CPU driver.
- **Drafter shape:** the drafter must be sliding-window. A full-attention drafter needs per-chunk
  injection during the prefill.
- **Selector and head:** the selector and top-k run on the host, with 3 MB of logits read back
  per draft. A GPU top-k would remove the download.
- **Draft length:** fixed at 7. An adaptive length (MTP's `n_dyn`) would cut wasted verify rows on
  low-acceptance text.
