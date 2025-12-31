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


def verus_minimize(code: str, interestingness_test: str, cores: int = 4, timeout: int = 30) -> dict:
    """
    Minimize Verus code using the creduce-based minimizer.
    
    Args:
        code: The Verus code to minimize
        interestingness_test: Type of minimizer ("invariant", "spec", "minimal", etc.)
        cores: Number of cores to use for minimization
        timeout: Verification timeout in seconds
    
    Returns:
        Dictionary with keys:
        - 'code': Minimized code as a string
        - 'status': 'success' if minimization completed, 'failed' otherwise
        - 'savings': Savings percentage as an integer (or None if not found)
    """
    minimizer_dir = Path("/Users/ameliakuang/Repos/verus-sft/chuyue_verus/source/tools/minimizers")
    minimize_script = minimizer_dir / "minimize.sh"
    
    if not minimize_script.exists():
        raise FileNotFoundError(f"Minimizer script not found at {minimize_script}")
    
    # Create a temporary file with the code
    with tempfile.NamedTemporaryFile(mode='w', suffix='.rs', delete=False, encoding='utf-8') as tmp_file:
        tmp_file.write(code)
        tmp_input = tmp_file.name
    
    try:
        # Run the minimizer script
        env = os.environ.copy()
        env["TIMEOUT"] = str(timeout)
        
        result = subprocess.run(
            [str(minimize_script), tmp_input, interestingness_test, str(cores)],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        
        # Parse output for status and savings
        output = result.stdout + result.stderr
        status = "success" if "Minimization Complete!" in output else "failed"
        
        # Extract savings ratio using regex
        savings = None
        if status == "success":
            savings_match = re.search(r"Savings:\s*(\d+)%", output)
        if savings_match:
            savings = int(savings_match.group(1))
        
        # The minimizer writes output to foo.rs in the minimizer directory
        minimized_file = minimizer_dir / "foo.rs"
        if minimized_file.exists():
            minimized_code = minimized_file.read_text(encoding="utf-8")
        else:
            # If foo.rs doesn't exist, return original code
            minimized_code = code
            if status == "success":
                status = "failed"  # Can't be successful if output file doesn't exist
        
        return {
            "code": minimized_code,
            "status": status,
            "savings": savings
        }
    except Exception as e:
        # Minimization failed, return original code
        return {
            "code": code,
            "status": "failed",
            "savings": None
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


# TODO: check with Livia - whether this verification logic is correct
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
    rust_files.sort()
    results: List[ExtractionResult] = []

    for idx, path in enumerate(rust_files):
        if limit is not None and idx >= limit:
            break
        result = attempt_extract(path, repo_url=repo_url, commit_sha=commit_sha, interestingness_test=interestingness_test)
        results.append(result)
        print(result.to_json())

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"
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
