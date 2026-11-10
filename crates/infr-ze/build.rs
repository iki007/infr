//! Compiles `kernels/*.cl` (OpenCL C) with Intel's offline compiler `ocloc` into native device
//! binaries in `OUT_DIR` (`<stem>.bin`), embedded by `src/kernels.rs` — the `glslc` step of
//! `infr-vulkan`, for Level Zero. The target defaults to the Arc Pro B70 (`bmg-g31`);
//! `INFR_ZE_TARGET` overrides it (any `ocloc -device` value). A kernel's first lines may carry
//! `// ocloc-options: ...` (e.g. the 256-register mode).
//!
//! Without `ocloc` on PATH the crate builds with no kernels: `cfg(infr_ze_kernels)` stays unset and
//! the GPU probes skip, so a machine without Intel's toolchain (CI) still builds the workspace.
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(infr_ze_kernels)");
    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-env-changed=INFR_ZE_TARGET");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let target = std::env::var("INFR_ZE_TARGET").unwrap_or_else(|_| "bmg-g31".to_string());
    if Command::new("ocloc").arg("--version").output().is_err() {
        println!("cargo:warning=infr-ze: ocloc not found, building without Level Zero kernels");
        return;
    }
    let mut srcs: Vec<PathBuf> = std::fs::read_dir("kernels")
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "cl"))
        .collect();
    srcs.sort();
    for src in srcs {
        println!("cargo:rerun-if-changed={}", src.display());
        let stem = src.file_stem().unwrap().to_string_lossy().into_owned();
        let text = std::fs::read_to_string(&src).unwrap();
        let mut options = String::from("-cl-std=CL3.0");
        for line in text.lines().take(8) {
            if let Some(o) = line.trim().strip_prefix("// ocloc-options:") {
                options.push(' ');
                options.push_str(o.trim());
            }
        }
        let st = Command::new("ocloc")
            .args(["compile", "-q", "-file"])
            .arg(&src)
            .args([
                "-device",
                &target,
                "-options",
                &options,
                "-output_no_suffix",
            ])
            .arg("-out_dir")
            .arg(&out)
            .args(["-output", &stem])
            .status()
            .expect("run ocloc");
        assert!(st.success(), "ocloc failed on {}", src.display());
        assert!(
            out.join(format!("{stem}.bin")).exists(),
            "ocloc wrote no {stem}.bin"
        );
    }
    println!("cargo:rustc-cfg=infr_ze_kernels");
}
