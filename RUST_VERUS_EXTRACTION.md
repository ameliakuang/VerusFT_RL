# Rust/Verus Code Extraction Method

Goal: extract **clean, dependency-free, compilable/verifiable** Rust/Verus snippets from large repositories so they can be used for SFT/RL without vendor-specific baggage.

**Objectives**

1. **Preserve semantic roles**: clearly separate executable code, specifications (`requires`/`ensures`/`decreases`/invariants),
   and proof blocks/lemmas so downstream datasets can tag `exec` vs. `spec` vs. `proof` content.
2. **Produce tiny, self-contained samples**: prefer minimized programs that verify (or consistently fail for negative examples)
   without external dependencies.
3. **Prefer the minimizer when available**: if the upstream project already works with the Verus minimizer, run it first to
   shrink candidates before manual extraction.

This method assumes you have access to the repo on disk and a working Verus toolchain (e.g., `verus --version`). It emphasizes **determinism**, **traceability**, and **safety** for downstream training.

## 1. Candidate Discovery

1. **Index the repo**: collect all `Cargo.toml` and `.rs` files, excluding build artifacts (`target/`, `tests/`, `examples/`, `benches/`, `docs/`, `vendor/`).
2. **Verus heuristics**: flag files containing any of the following tokens (fast `rg` queries):
   - `verus!`, `#[verus::]`, `requires`, `ensures`, `decreases`, `invariant`, `ghost`, `proof`, `spec`, `exec`, `reveal`, `opens_invariants`.
3. **Rank candidates** by a simple score: `verus_token_count + spec_keyword_count` to prioritize likely verification code.

## 2. Dependency Tracing

1. **Parse crate graph**: run `cargo metadata --no-deps --format-version 1` at the repo root to obtain the workspace members and dependency graph.
2. **Local dependency check**: for each candidate file, extract `use` statements and match them to the crate graph. Prefer files that only depend on:
   - `std`, `core`, or Verus prelude modules (`vstd`, `verus::prelude::*`).
   - Sibling modules within the same crate.
3. **External crate filter**: if the file pulls non-std/non-Verus crates, attempt to inline minimal definitions (if trivial) or **skip** to keep the dataset dependency-free.

## 3. Self-Contained Snippet Construction

1. **Module closure**: collect the file plus any local modules it references (e.g., `mod utils; use crate::utils::...`). Inline those modules into a single buffer while preserving order.
2. **Feature stripping**: remove `#[cfg(test)]`, `#[cfg(feature = ...)]`, and test-only functions to avoid non-compiling code.
3. **Minimal crate wrapper**: wrap the combined code in a synthetic crate:
   ```text
   /tmp/extract-<hash>/
     Cargo.toml   # name = "verus_extract_<hash>", edition = 2021
     src/lib.rs   # contains `verus! { ... }` wrapper if not already present
   ```
4. **Deterministic formatting**: run `rustfmt` on the snippet; for Verus blocks, keep the original indentation to avoid altering proof scripts unexpectedly.

## 4. Verification/Compilation Loop

1. **Fast syntax check**: `cargo check --offline` for plain Rust or `verus --verify --crate-type=lib src/lib.rs` for Verus code.
2. **Timeout guard**: cap verification to ~30s per snippet to avoid hanging jobs.
3. **Error categorization** (reuse `verus_verification.categorize_verus_error` if available): syntax, mode, pre/postcondition, invariant, termination, timeout, unknown.
4. **Acceptance rule**: keep snippets that **compile/verify cleanly with zero external dependencies**; discard otherwise.

## 5. Provenance and Metadata

Emit a manifest entry per accepted snippet (JSON or Parquet):

```json
{
  "source_path": "crates/foo/src/bar.rs",
  "crate": "foo",
  "sha": "<git-sha>",
  "rank_score": 12,
  "verus_tokens": 7,
  "dependencies": ["std", "vstd"],
  "status": "verified",  // or "compiled"
  "verify_time_ms": 1830,
  "notes": "inlined mod math" 
}
```

## 6. Deduplication and Quality Filters

- **Near-duplicate removal**: hash normalized ASTs or use MinHash over tokenized code to drop duplicates.
- **Length limits**: enforce sane bounds (e.g., 5–400 lines) to avoid trivial or enormous proofs.
- **Spec completeness**: require at least one of `requires`/`ensures` or a loop invariant for inclusion.

## 7. Batch Execution Plan

1. `python verus_code_extraction.py --repo /path/to/repo --out ./extracted_snippets`
2. Inspect the manifest for failures; iterate on the dependency filter thresholds before running at scale.
3. Archive accepted snippets as text plus metadata; store failures separately for future repair tasks.

## 8. Safety/Compliance Notes

- Do **not** pull network dependencies during extraction (`cargo check --offline`).
- Avoid copying license-incompatible code; honor repo licenses in downstream datasets.
- Keep a reproducible log (commands, tool versions, git SHAs) for every batch run.

## Reference Implementation Skeleton

See `verus_code_extraction.py` for a minimal, scriptable starting point.
