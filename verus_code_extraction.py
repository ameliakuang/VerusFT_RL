"""
Minimal, scriptable scaffold for extracting dependency-free Verus/Rust snippets
from larger repositories. The goal is to keep extraction logic deterministic and
traceable; adapt the functions to your environment and toolchain paths.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from utils import eval_verus

# Heuristic tokens that signal Verus/verification-oriented code
VERUS_TOKENS = {
    "verus!",
    "#[verus::",
    "requires",
    "ensures",
    "decreases",
    "invariant",
    "ghost",
    "proof",
    "spec",
    "exec",
    "reveal",
    "opens_invariants",
}


@dataclass
class ExtractionResult:
    source_path: Path
    status: str
    message: str
    code: Optional[str] = None
    minimized_code: Optional[str] = None
    verus_tokens: int = 0
    original_LOC: Optional[int] = 0
    minimized_LOC: Optional[int] = 0
    reduction_ratio: Optional[float] = 0.0
    minimum_verifiable: Optional[bool] = False
    dependencies: List[str] = field(default_factory=list)
    self_contained: Optional[bool] = False
    verify_time_ms: Optional[int] = None
    repo_url: str = ""
    commit_sha: str = ""
    minimized_status: Optional[str] = None
    segments: dict[str, str] = field(default_factory=dict)  # {"exec": "...", "spec": "...", "proof": "..."}

    def to_json(self) -> str:
        payload = {
            "source_path": str(self.source_path),
            "status": self.status,
            "message": self.message,
            "code": self.code,
            "minimized_code": self.minimized_code,
            "dependencies": self.dependencies,
            "verus_tokens": self.verus_tokens,
            "original_LOC": self.original_LOC,
            "minimized_LOC": self.minimized_LOC,
            "reduction_ratio": self.reduction_ratio,
            "minimum_verifiable": self.minimum_verifiable,
            "dependencies": self.dependencies,
            "self_contained": self.self_contained,
            "verify_time_ms": self.verify_time_ms,
            "repo_url": self.repo_url,
            "commit_sha": self.commit_sha,
            "minimized_status": self.minimized_status,
            "segments": self.segments,
        }
        return json.dumps(payload, ensure_ascii=False)


def find_rust_files(repo_root: Path) -> List[Path]:
    ignore_dirs = {"target", "tests", "benches", "docs", "vendor", ".git"}
    rust_files = []
    for path in repo_root.rglob("*.rs"):
        if any(part in ignore_dirs for part in path.parts):
            continue
        rust_files.append(path)
    return rust_files


def count_lines(path: Path) -> int:
    """Count the number of lines in a file."""
    try:
        with path.open(encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def contains_verus_tokens(text: str) -> bool:
    return any(token in text for token in VERUS_TOKENS)


def score_verus_tokens(text: str) -> int:
    return sum(text.count(token) for token in VERUS_TOKENS)


def load_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def get_git_info(repo_root: Path) -> tuple[str, str]:
    """
    Get the origin URL and current commit SHA from a git repository.
    Assumes the directory is a valid git repository with origin remote configured.
    """
    repo_url_result = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    repo_url = repo_url_result.stdout.strip()
    
    commit_sha_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    commit_sha = commit_sha_result.stdout.strip()
    
    return repo_url, commit_sha


def extract_local_dependencies(text: str) -> List[str]:
    pattern = re.compile(r"^use\s+([\w:]+)", re.MULTILINE)
    deps = []
    for match in pattern.finditer(text):
        deps.append(match.group(1))
    return deps


def score_snapshot(code: str, original_lines: int) -> float:
    """
    Score a snapshot based on how good it is for training.
    Higher score = better snapshot.
    
    Criteria:
    - Prefer 30-70% reduction (good balance between size and content)
    - Prefer code with semantic content (functions, structs, etc.)
    - Penalize code that's too small (< 5 lines or < 5% of original)
    - Penalize code that's too large (> 90% of original - not much reduction)
    """
    lines = len(code.splitlines())
    if lines == 0:
        return -1000.0
    
    reduction_pct = (1 - lines / original_lines) * 100
    
    # Base score from reduction percentage (prefer 30-70% reduction)
    if 30 <= reduction_pct <= 70:
        score = 100.0
    elif 20 <= reduction_pct < 30 or 70 < reduction_pct <= 80:
        score = 80.0
    elif 10 <= reduction_pct < 20 or 80 < reduction_pct <= 90:
        score = 60.0
    else:
        score = 40.0
    
    # Penalize if too small (likely lost semantic meaning)
    if lines < 5 or reduction_pct > 95:
        score -= 50.0
    
    # Penalize if too large (not much reduction)
    if reduction_pct < 10:
        score -= 30.0
    
    # Bonus for semantic content
    semantic_keywords = ["fn ", "struct ", "enum ", "impl ", "trait ", "spec ", "proof "]
    keyword_count = sum(code.count(kw) for kw in semantic_keywords)
    score += min(keyword_count * 5, 20)  # Up to 20 bonus points
    
    # Bonus for having function definitions (most important)
    if re.search(r'\bfn\s+\w+\s*\(', code):
        score += 15.0
    
    return score


def select_best_snapshot(snapshot_dir: Path, original_lines: int) -> Optional[Path]:
    """
    Select the best snapshot from available snapshots.
    Returns the path to the best snapshot, or None if no snapshots found.
    """
    snapshot_files = sorted(snapshot_dir.glob("foo.rs.snapshot_*"))
    
    if not snapshot_files:
        return None
    
    best_snapshot = None
    best_score = -float('inf')
    
    for snapshot_file in snapshot_files:
        try:
            snapshot_code = snapshot_file.read_text(encoding="utf-8")
            score = score_snapshot(snapshot_code, original_lines)
            
            if score > best_score:
                best_score = score
                best_snapshot = snapshot_file
        except Exception:
            continue
    
    return best_snapshot


def verus_minimize(code: str, interestingness_test: str, cores: int = 4, timeout: int = 30) -> dict:
    """
    Minimize Verus code using the creduce-based minimizer with snapshots.
    
    Args:
        code: The Verus code to minimize
        interestingness_test: Type of minimizer ("invariant", "spec", "minimal", etc.)
        cores: Number of cores to use for minimization
        timeout: Verification timeout in seconds
    
    Returns:
        Dictionary with keys:
        - 'code': Minimized code as a string (best snapshot selected)
        - 'status': 'success' if minimization completed, 'failed' otherwise
        - 'savings': Savings percentage as an integer (or None if not found)
        - 'snapshot_used': Path to the snapshot file used (or None)
        - 'all_snapshots': List of all snapshot paths (for manual review)
    """
    minimizer_dir = Path("/Users/ameliakuang/Repos/verus-sft/chuyue_verus/source/tools/minimizers")
    minimize_script = minimizer_dir / "minimize_with_snapshots.sh"
    snapshot_dir = minimizer_dir / "snapshots"
    
    if not minimize_script.exists():
        raise FileNotFoundError(f"Minimizer script not found at {minimize_script}")
    
    # Create a temporary file with the code
    with tempfile.NamedTemporaryFile(mode='w', suffix='.rs', delete=False, encoding='utf-8') as tmp_file:
        tmp_file.write(code)
        tmp_input = tmp_file.name
    
    original_lines = len(code.splitlines())
    
    try:
        # Run the minimizer script with snapshots
        env = os.environ.copy()
        env["TIMEOUT"] = str(timeout)
        
        # Use snapshot interval of 30 seconds
        cmd = [str(minimize_script), tmp_input, interestingness_test, str(cores), "30"]
        print(f"Running minimizer: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,  # Don't fail if minimization has issues - we'll check returncode
            env=env,
            timeout=timeout * 60,  # Add timeout back
        )
        
        # Parse output for status
        output = result.stdout + result.stderr

        # Debug: print errors if script failed
        if result.returncode != 0:
            print(f"Warning: Minimizer script exited with code {result.returncode}")
            print(f"Stdout: {result.stdout[:1000]}")
            print(f"Stderr: {result.stderr[:1000]}")
        
        status = "success" if "Minimization Complete!" in output else "failed"
        
        # Try to find and select best snapshot
        minimized_code = code
        snapshot_used = None
        all_snapshots = []
        savings = None
        
        if snapshot_dir.exists():
            all_snapshots = sorted(snapshot_dir.glob("foo.rs.snapshot_*"))
            
            if all_snapshots:
                # Select best snapshot based on scoring
                best_snapshot = select_best_snapshot(snapshot_dir, original_lines)
                
                if best_snapshot:
                    minimized_code = best_snapshot.read_text(encoding="utf-8")
                    snapshot_used = best_snapshot
                    final_lines = len(minimized_code.splitlines())
                    savings = int((1 - final_lines / original_lines) * 100) if original_lines > 0 else 0
                    status = "success"
        
        # Fallback to final foo.rs if no good snapshot found
        if not snapshot_used:
            minimized_file = minimizer_dir / "foo.rs"
            if minimized_file.exists():
                minimized_code = minimized_file.read_text(encoding="utf-8")
                final_lines = len(minimized_code.splitlines())
                savings = int((1 - final_lines / original_lines) * 100) if original_lines > 0 else 0
                
                # Only mark as success if we got meaningful reduction
                if savings and savings > 5:
                    status = "success"
        
        return {
            "code": minimized_code,
            "status": status,
            "savings": savings,
            "snapshot_used": str(snapshot_used) if snapshot_used else None,
            "all_snapshots": [str(s) for s in all_snapshots],
        }
    except subprocess.TimeoutExpired:
        # Minimization timed out
        print(f"Warning: Minimization timed out after {timeout * 60} seconds")
        return {
            "code": code,
            "status": "failed",
            "savings": None,
            "snapshot_used": None,
            "all_snapshots": [],
        }
    except subprocess.TimeoutExpired:
        # Minimization timed out
        print(f"Warning: Minimization timed out after {timeout * 60} seconds")
        return {
            "code": code,
            "status": "failed",
            "savings": None,
            "snapshot_used": None,
            "all_snapshots": [],
        }
    except Exception as e:
        # Minimization failed, return original code
        print(f"Warning: Minimization failed with error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "code": code,
            "status": "failed",
            "savings": None,
            "snapshot_used": None,
            "all_snapshots": [],
        }
    finally:
        # Clean up temporary input file
        try:
            os.unlink(tmp_input)
        except Exception:
            pass


def segment_verus_code(code: str) -> dict[str, str]:
    """
    Segment Verus code into exec, spec, and proof zones.
    
    Returns a dictionary with keys "exec", "spec", and "proof" containing
    the code segments for each mode.
    """
    exec_parts = []
    spec_parts = []
    proof_parts = []
    spec_clauses = []
    
    lines = code.split('\n')
    
    # Extract spec functions (spec fn or #[spec] fn)
    spec_function_pattern = re.compile(r'(?:#\[spec\]\s*|spec\s+)fn\s+(\w+)', re.MULTILINE)
    for match in spec_function_pattern.finditer(code):
        # Find the function body
        func_name = match.group(1)
        escaped_name = re.escape(func_name)
        start_pos = match.start()
        # Try to find the function body (simplified - looks for next { to })
        pattern = 'spec\\s+fn\\s+' + escaped_name + '.*?\\{'
        func_match = re.search(pattern, code[start_pos:], re.DOTALL)
        if func_match:
            spec_parts.append(code[start_pos:start_pos + func_match.end()])
    
    # Extract proof functions (proof fn or #[proof] fn)
    proof_function_pattern = re.compile(r'(?:#\[proof\]\s*|proof\s+)fn\s+(\w+)', re.MULTILINE)
    for match in proof_function_pattern.finditer(code):
        func_name = match.group(1)
        escaped_name = re.escape(func_name)
        start_pos = match.start()
        pattern = 'proof\\s+fn\\s+' + escaped_name + '.*?\\{'
        func_match = re.search(pattern, code[start_pos:], re.DOTALL)
        if func_match:
            proof_parts.append(code[start_pos:start_pos + func_match.end()])
    
    # Extract proof blocks (proof { ... })
    proof_block_pattern = re.compile(r'proof\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', re.DOTALL)
    for match in proof_block_pattern.finditer(code):
        proof_parts.append(match.group(0))
    
    # Extract exec functions (default or #[exec] fn)
    exec_function_pattern = re.compile(r'(?:#\[exec\]\s*|^|\s)(?:pub\s+)?fn\s+(\w+)', re.MULTILINE)
    for match in exec_function_pattern.finditer(code):
        func_name = match.group(1)
        # Skip if it's actually a spec or proof function
        before_match = code[max(0, match.start()-50):match.start()]
        if 'spec' in before_match or 'proof' in before_match:
            continue
        escaped_name = re.escape(func_name)
        start_pos = match.start()
        pattern = 'fn\\s+' + escaped_name + '.*?\\{'
        func_match = re.search(pattern, code[start_pos:], re.DOTALL)
        if func_match:
            exec_parts.append(code[start_pos:start_pos + func_match.end()])
    
    # Extract spec clauses (requires, ensures, invariant)
    for line in lines:
        if re.search(r'\b(requires|ensures|invariant|decreases)\b', line):
            spec_clauses.append(line)
    
    # If no specific segments found, classify entire code as exec
    if not exec_parts and not spec_parts and not proof_parts:
        exec_parts.append(code)
    
    return {
        "exec": '\n\n'.join(exec_parts) if exec_parts else "",
        "spec": '\n\n'.join(spec_parts + spec_clauses) if (spec_parts or spec_clauses) else "",
        "proof": '\n\n'.join(proof_parts) if proof_parts else "",
    }


def build_temp_crate(snippet: str) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="verus_extract_"))
    src_dir = temp_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    lib_rs = src_dir / "lib.rs"
    if "verus!" not in snippet:
        snippet = f"verus! {{\n{snippet}\n}}\n"
    lib_rs.write_text(snippet, encoding="utf-8")

    cargo_toml = temp_dir / "Cargo.toml"
    cargo_toml.write_text(
        "\n".join(
            [
                "[package]",
                "name = \"verus_extract\"",
                "version = \"0.1.0\"",
                "edition = \"2021\"",
                "",
                "[dependencies]",
                "verus = \"*\"",
            ]
        ),
        encoding="utf-8",
    )
    return temp_dir


def verify_snippet(temp_crate: Path, timeout: int = 30) -> subprocess.CompletedProcess:
    cmd = ["verus","--crate-type=lib", str(temp_crate / "src" / "lib.rs")]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)

def check_dependencies(dependencies: list) -> bool:
    """Check if dependencies only include std, core, or vstd."""
    if not dependencies:
        return True
    
    allowed_roots = {
        "std",
        "core",
        "vstd",
        "verus_builtin",
        "verus_builtin_macros",
    }

    for dep in dependencies:
        # Normalize: strip trailing :: and whitespace
        dep = dep.strip().rstrip(":")
        if not dep:
            continue

        # Extract crate root
        root = dep.split("::", 1)[0]

        if root not in allowed_roots:
            return False

    return True

def attempt_extract(path: Path, repo_url: str, commit_sha: str, interestingness_test: str = "minimal") -> ExtractionResult:
    text = load_file(path)
    if not contains_verus_tokens(text):
        segments = segment_verus_code(text)
        return ExtractionResult(
            path, 
            "skipped", 
            "no_verus_tokens", 
            code=text,
            minimized_code=text,
            verus_tokens=0,
            original_LOC=len(text.splitlines()),
            minimized_LOC=len(text.splitlines()),
            reduction_ratio=0.0,
            minimum_verifiable=None,
            dependencies=[],
            self_contained=None,
            repo_url=repo_url,
            commit_sha=commit_sha,
            minimized_status=None,
            segments=segments,
        )

    score = score_verus_tokens(text)
    deps = extract_local_dependencies(text)
    # temp_crate = build_temp_crate(text)
    minimized_code = text  # Default to original code
    minimized_status = None
    segments = {}
    
    try:
        start_time = time.perf_counter()
        # proc = verify_snippet(temp_crate)
        verus_result = eval_verus(text)
        verus_succeed = verus_result["verus_succeed"]
        verified_count = verus_result["verified_count"]
        error_count = verus_result["error_count"]
        verus_errors = verus_result["verus_errors"]
        compilation_error = verus_result["compilation_error"]
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        
        if verus_succeed:
            # Code verifies, attempt minimization
            try:
                minimize_result = verus_minimize(text, interestingness_test)
                minimized_code = minimize_result["code"]
                # Map "success" to "succeeded" for consistency with existing code
                minimize_status = minimize_result["status"]
                minimized_status = "succeeded" if minimize_status == "success" else "failed"
                savings = minimize_result.get("savings")
                
                # Segment the minimized code
                segments = segment_verus_code(minimized_code)
            except Exception as e:
                minimized_status = "failed"
                minimized_code = text  # Fall back to original
                segments = segment_verus_code(text)
                savings = None
            
            # Build message with savings if available
            message = f"verified with minimized code with {interestingness_test}, original score: {score}"
            if savings is not None:
                message += f", savings: {savings}%"
            
            return ExtractionResult(
                path,
                "verified",
                message,
                code=text,
                minimized_code=minimized_code,
                verus_tokens=score,
                original_LOC=len(text.splitlines()),
                minimized_LOC=len(minimized_code.splitlines()),
                reduction_ratio=len(minimized_code.splitlines()) / len(text.splitlines()) if len(text.splitlines()) > 0 else None,
                minimum_verifiable=None,
                self_contained=check_dependencies(deps),
                dependencies=deps,
                verify_time_ms=elapsed_ms,
                repo_url=repo_url,
                commit_sha=commit_sha,
                minimized_status=minimized_status,
                segments=segments,
            )
        
        # Code doesn't verify - still segment original code
        segments = segment_verus_code(text)
        return ExtractionResult(
            path,
            "failed",
            verus_errors,
            code=text,
            minimized_code=minimized_code,
            original_LOC=len(text.splitlines()),
            minimized_LOC=len(minimized_code.splitlines()),
            reduction_ratio=len(minimized_code.splitlines()) / len(text.splitlines()) if len(text.splitlines()) > 0 else None,
            minimum_verifiable=False,
            self_contained=check_dependencies(deps),
            dependencies=deps,
            verus_tokens=score,
            verify_time_ms=elapsed_ms,
            repo_url=repo_url,
            commit_sha=commit_sha,
            minimized_status=minimized_status,
            segments=segments,
        )
    except subprocess.TimeoutExpired:
        segments = segment_verus_code(text)
        return ExtractionResult(
            path, 
            "timeout", 
            "verification_timeout", 
            code=text,
            minimized_code=minimized_code, 
            verus_tokens=score,
            dependencies=deps,
            repo_url=repo_url,
            commit_sha=commit_sha,
            minimized_status=minimized_status,
            segments=segments,
        )
    # finally:
    #     shutil.rmtree(temp_crate, ignore_errors=True)


def main(repo: Path, out_dir: Path, limit: Optional[int] = None, interestingness_test: str = "minimal") -> None:
    # Get git repository information
    repo_url, commit_sha = get_git_info(repo)
    print(f"Repository URL: {repo_url}")
    print(f"Commit SHA: {commit_sha}")
    
    rust_files = find_rust_files(repo)
    print(f"Found {len(rust_files)} Rust files")
    
    # Filter files by line count (5-400 lines)
    rust_files = [f for f in rust_files if 5 <= count_lines(f) <= 400]
    print(f"Filtered to {len(rust_files)} files with 5-400 lines")
    rust_files.sort()
    results: List[ExtractionResult] = []

    for idx, path in enumerate(rust_files):
        if limit is not None and idx >= limit:
            break
        result = attempt_extract(path, repo_url=repo_url, commit_sha=commit_sha, interestingness_test=interestingness_test)
        results.append(result)
        print(result.to_json())

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest_3.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for item in results:
            f.write(item.to_json() + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract dependency-free Verus snippets from a repository")
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Path to the repository root")
    parser.add_argument("--out", type=Path, default=Path("./extracted_snippets"), help="Output directory for manifest")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of files to process")
    parser.add_argument("--interestingness_test", type=str, default="minimal", choices=['minimal', 'invariant', 'spec'], help="Interestingness test for minimization")
    args = parser.parse_args()

    main(args.repo, args.out, args.limit, args.interestingness_test)
