# Verus Code Extraction Tool

A tool for extracting dependency-free Verus/Rust snippets from larger repositories. `verus_code_extraction.py` identifies Verus code, verifies it, minimizes it while preserving verification properties, and generates structured datasets.

## Algorithm

The extraction process follows these steps:

1. **File Discovery**: Scans repository for `.rs` files (50-500 lines)
2. **Token Detection**: Identifies Verus code by scanning for verification-specific tokens (`requires`, `ensures`, `invariant`, `proof`, `verus!`, etc.)
3. **Verification**: Runs Verus verifier on the code to ensure it's valid
4. **Minimization** (if verified): Uses creduce-based minimizer to reduce code size while preserving verification properties
5. **Segmentation**: Separates code into exec (executable), spec (specification), and proof zones
6. **Quality Assessment**: Computes metrics (readability scores, reduction ratios, self-containment checks)
7. **Export**: Writes results to JSONL manifest file

## Usage

### Running the Extraction Tool

```bash
python verus_code_extraction.py --repo /path/to/repo --out ./extracted_snippets
```

**Command-Line Options:**
- `--repo PATH`: Path to the repository root (default: current directory)
- `--out PATH`: Output directory for manifest files (default: `./extracted_snippets`)
- `--limit N`: Limit the number of files to process
- `--interestingness_test TYPE`: Type of minimization test (`minimal`, `invariant`, `spec`)
- `--continue`: Resume processing from existing manifest

### Running the Analysis Tool

The `analysis.py` script processes the extracted manifest files to generate a CSV summary and statistics:

```bash
python analysis.py --base_dir ./extracted_snippets
```

This generates:
- `analysis_summary.csv`: Tabular summary of all extractions with metrics
- Console output: Verification rates, minimization success, quality metrics

## Extracted Snippets Structure

The tool generates output in the `extracted_snippets` directory with the following structure:

```
extracted_snippets/
├── autoverus_examples/
│   └── manifest.jsonl
├── vericoding_benchmarks_verus/
│   └── manifest.jsonl
├── verus_examples/
│   └── manifest.jsonl
└── analysis_summary.csv
```

Each `manifest.jsonl` file contains one JSON object per line with the following structure:

```json
{
  "id": "filename_abc12345",
  "original_code": "...",
  "minimized_code": "...",
  "all_snapshots": ["...", "..."],
  "metadata": {
    "provenance": {
      "source_repo": "https://github.com/user/repo",
      "file_path": "/absolute/path/to/file.rs",
      "commit_sha": "abc123..."
    },
    "verification": {
      "status": "verified",
      "error_type": "none",
      "verify_time_ms": 1234,
      "minimize_time_ms": 5678,
      "minimum_verifiable": true,
      "minimized_status": "succeeded",
      "verus_errors": []
    },
    "quality": {
      "original_LOC": 150,
      "minimized_LOC": 75,
      "reduction_ratio": 0.5,
      "readability_score": 0.85,
      "self_contained": true,
      "dependencies": ["vstd::prelude::*"],
      "complexity_verus_tokens": 2,
      "has_meaningful_spec": True,
    },
    "labeling": {
      "has_invariant": true,
      "has_proof_block": false,
      "requires_count": 2,
      "ensures_count": 1,
      "segments": {
        "exec": "...",
        "spec": "...",
        "proof": "..."
      }
    }
  }
}
```

The `analysis.py` script reads all `manifest.jsonl` files from the subdirectories and generates a consolidated `analysis_summary.csv` file with extracted metrics for analysis.

## Dataset Descriptions

The current dataset contains extracted snippets from three repositories:

- **autoverus_examples**: 15 samples extracted from [microsoft/verus-proof-synthesis/autoverus/examples](https://github.com/microsoft/verus-proof-synthesis/tree/main/autoverus/examples)
- **vericoding_benchmarks_verus**: 420 samples extracted from [Beneficial-AI-Foundation/vericoding/benchmarks/verus](https://github.com/Beneficial-AI-Foundation/vericoding/tree/main/benchmarks/verus)
- **verus_examples**: 103 samples extracted from [verus-lang/verus/examples](https://github.com/verus-lang/verus/tree/main/examples)

**Total**: 538 samples across all repositories.