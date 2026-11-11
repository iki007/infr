//! DFlash2 speculative decoding on a real target + drafter pair (GPU, opt-in).
//!
//! ```text
//! INFR_DFLASH_TARGET=/path/Qwen3.8-27B-UD-Q4_K_M.gguf \
//! INFR_DFLASH_DRAFT=/path/Qwen3.8-27B-DFlash2-...gguf INFR_DEV=Vulkan1 \
//!   cargo test --release -p infr-llama --test dflash -- --ignored --nocapture
//! ```
//!
//! Greedy DFlash commits the target's own argmax stream, so its text must match plain greedy
//! decoding — up to the near-tie caveat MTP documents (the verify batch and the one-row decode
//! round differently; see `infr_llama::mtp::mtp_enabled`).

use std::path::PathBuf;

fn paths() -> Option<(PathBuf, PathBuf)> {
    let t = std::env::var_os("INFR_DFLASH_TARGET")?;
    let d = std::env::var_os("INFR_DFLASH_DRAFT")?;
    Some((PathBuf::from(t), PathBuf::from(d)))
}

const PROMPT: &str = "Write a Python function that parses an ISO-8601 date string into a \
                      datetime object without using any libraries, with input validation and \
                      a few doctests.";

#[test]
#[ignore = "needs a GPU and INFR_DFLASH_TARGET / INFR_DFLASH_DRAFT"]
fn dflash_greedy_matches_plain_greedy() {
    let Some((target, draft)) = paths() else {
        eprintln!("skipped: set INFR_DFLASH_TARGET and INFR_DFLASH_DRAFT");
        return;
    };
    let n: usize = std::env::var("INFR_DFLASH_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);
    // `INFR_PROF_STAGES=1` reports through `tracing`; show it on stderr.
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .try_init();
    let mut cfg = infr_llama::EngineConfig::default();
    cfg.device.dev = std::env::var("INFR_DEV").ok();
    cfg.prof.stages = std::env::var("INFR_PROF_STAGES").is_ok_and(|v| v == "1");
    cfg.prof.ops = std::env::var("INFR_PROF_OPS").is_ok_and(|v| v == "1");
    cfg.prof.op_shapes = std::env::var("INFR_PROF_OP_SHAPES").is_ok_and(|v| v == "1");
    // Sampled decoding (Qwen's recommended temp/top-k/top-p): the speculative-sampling accept path.
    let temp: Option<f32> = std::env::var("INFR_DFLASH_TEMP")
        .ok()
        .and_then(|v| v.parse().ok());
    if let Some(t) = temp {
        cfg.sampling.temp = t;
        cfg.sampling.top_k = 20;
        cfg.sampling.top_p = 0.95;
        cfg.sampling.seed = Some(1);
    }
    // Speed work only: skip the plain-greedy reference run (and the comparison).
    let skip_plain = std::env::var("INFR_DFLASH_SKIP_PLAIN").is_ok_and(|v| v == "1");
    let model = infr_llama::SeamModel::load_with(&target, None, std::sync::Arc::new(cfg))
        .expect("target load");
    let drafter = std::sync::Arc::new(
        infr_llama::dflash::DflashDrafter::open(&draft, model.config()).expect("drafter"),
    );
    let prompt = model.render_chat(PROMPT).expect("render chat");
    if let Ok(path) = std::env::var("INFR_DFLASH_DUMP_PROMPT") {
        std::fs::write(path, &prompt).expect("dump prompt");
    }

    let mut plain = String::new();
    let ps = if skip_plain {
        infr_llama::GenStats {
            n_prompt: 0,
            prompt_secs: 0.0,
            n_gen: 1,
            decode_secs: 1.0,
        }
    } else {
        let mut sess = model.vulkan_session(8192).expect("target session");
        model
            .generate_vulkan_session(&mut sess, &prompt, n, None, |p| plain.push_str(p))
            .expect("plain greedy")
    };
    println!(
        "plain : {} tokens, {:.2} t/s decode",
        ps.n_gen,
        ps.n_gen as f64 / ps.decode_secs
    );

    let mut spec = String::new();
    let (ds, tm) =
        infr_llama::dflash::generate_dflash_spec_vulkan(&model, &drafter, &prompt, n, |p| {
            spec.push_str(p)
        })
        .expect("dflash decode");
    println!(
        "dflash: {} tokens, {:.2} t/s decode ({:.2}x), {} cycles, alpha {:.3}, {:.2} tokens/cycle; \
         draft {:.0} ms, verify {:.0} ms, inject {:.0} ms, rollback {:.0} ms, {} reprimes {:.0} ms",
        ds.n_gen,
        ds.n_gen as f64 / ds.decode_secs,
        (ds.n_gen as f64 / ds.decode_secs) / (ps.n_gen as f64 / ps.decode_secs),
        tm.cycles,
        tm.alpha(),
        tm.tokens_per_cycle(),
        tm.draft_secs * 1e3,
        tm.verify_secs * 1e3,
        tm.inject_secs * 1e3,
        tm.other_secs * 1e3,
        tm.reprimes,
        tm.reprime_secs * 1e3,
    );
    if skip_plain || temp.is_some() {
        return;
    }
    if spec != plain {
        let at = spec
            .chars()
            .zip(plain.chars())
            .take_while(|(a, b)| a == b)
            .count();
        println!("--- diverged at char {at}\n--- plain:\n{plain}\n--- dflash:\n{spec}");
    }
    assert_eq!(spec, plain, "DFlash greedy diverged from plain greedy");
}
