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
from utils import eval_verus, VerusError

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
    verus_errors: Optional[List[VerusError]] = None
    repo_url: str = ""
    commit_sha: str = ""
    repo_root: Optional[Path] = None  # Repository root path for computing relative paths
    minimized_status: Optional[str] = None
    segments: dict[str, str] = field(default_factory=dict)  # {"exec": "...", "spec": "...", "proof": "..."}
    minimize_time_ms: Optional[int] = None  # Time taken for minimization
    quality_score: Optional[float] = None  # Readability/quality score (0.0-1.0)
    is_meaningful: Optional[bool] = None  # Has requires/ensures/invariant/proof
    meets_criteria: Optional[bool] = None  # Meets all dataset inclusion criteria

    def to_json(self) -> str:
        """Convert to structured JSON format, only including fields with values."""
        code_to_use = self.minimized_code if self.minimized_code else self.code or ""
        
        # Generate ID from source path
        source_name = self.source_path.stem if self.source_path else "unknown"
        extraction_id = f"{source_name}_{self.commit_sha[:8]}" if self.commit_sha else source_name
        
        # Get error type
        if self.status == "verified":
            error_type = "none"
        elif self.status == "failed":
            msg_lower = self.message.lower() if self.message else ""
            if "precondition" in msg_lower:
                error_type = "precondition"
            elif "postcondition" in msg_lower:
                error_type = "postcondition"
            elif "invariant" in msg_lower:
                error_type = "invariant"
            else:
                error_type = "other"
        elif self.status == "timeout":
            error_type = "timeout"
        else:
            error_type = "unknown"
        
        # Count spec elements
        requires_count = len(re.findall(r'\brequires\s*\(', code_to_use)) if code_to_use else 0
        ensures_count = len(re.findall(r'\bensures\s*\(', code_to_use)) if code_to_use else 0
        has_invariant = bool(re.search(r'\binvariant\s*\(', code_to_use)) if code_to_use else False
        has_proof_block = bool(re.search(r'\bproof\s*\{', code_to_use)) if code_to_use else False

        # Build metadata sections, only including fields with values
        metadata = {}
        
        # Provenance
        provenance = {}
        provenance["source_repo"] = self.repo_url
        
        # Use absolute file path (simpler and more reliable)
        if self.source_path:
            provenance["file_path"] = str(self.source_path.resolve())
        
        provenance["commit_sha"] = self.commit_sha
        if provenance:
            metadata["provenance"] = provenance
        
        # Verification
        verification = {"status": self.status}
        if error_type != "none":
            verification["error_type"] = error_type
        if self.verify_time_ms is not None:
            verification["verify_time_ms"] = self.verify_time_ms
        if self.minimize_time_ms is not None:
            verification["minimize_time_ms"] = self.minimize_time_ms
        if self.verus_errors is not None:
            # Convert VerusError objects to JSON-serializable format
            verification["verus_errors"] = [
                {
                    "error_type": err.error.name,
                    "error_text": err.error_text,
                    "message": str(err),
                    "spans": [
                        {
                            "file_name": span.fname,
                            "lines": list(span.lines),
                            "label": span.strlabel,
                            "text": [
                                {"text": t.text, "highlight_start": t.hl_start, "highlight_end": t.hl_end}
                                for t in span.text
                            ]
                        }
                        for span in err.trace
                    ]
                }
                for err in self.verus_errors
            ]
        verification['minimum_verifiable'] = self.minimum_verifiable
        verification['minimized_status'] = self.minimized_status
        metadata["verification"] = verification
        
        # Quality
        quality = {}
        quality["original_LOC"] = self.original_LOC
        quality['minimized_LOC'] = self.minimized_LOC
        
        if self.self_contained is not None:
            quality["self_contained"] = self.self_contained
        if self.dependencies:
            quality["dependencies"] = self.dependencies
        if self.verus_tokens > 0:
            quality["complexity_verus_tokens"] = self.verus_tokens
        if self.is_meaningful is not None:
            quality["has_meaningful_spec"] = self.is_meaningful
        if self.quality_score is not None:
            quality["readability_score"] = self.quality_score
        if self.reduction_ratio is not None:
            quality["reduction_ratio"] = self.reduction_ratio
        if quality:
            metadata["quality"] = quality
        
        # Labeling
        labeling = {}
        if has_invariant:
            labeling["has_invariant"] = True
        if has_proof_block:
            labeling["has_proof_block"] = True
        if requires_count > 0:
            labeling["requires_count"] = requires_count
        if ensures_count > 0:
            labeling["ensures_count"] = ensures_count
        
        # Only include segments with content
        segments = {}
        if self.segments.get("exec"):
            segments["exec"] = self.segments["exec"]
        if self.segments.get("spec"):
            segments["spec"] = self.segments["spec"]
        if self.segments.get("proof"):
            segments["proof"] = self.segments["proof"]
        if segments:
            labeling["segments"] = segments
        
        if labeling:
            metadata["labeling"] = labeling
        
        # Build payload
        payload = {
            "id": extraction_id,
            "original_code": self.code,
            "minimized_code": self.minimized_code,
            "metadata": metadata
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


def is_meaningful_verus_code(code: str) -> bool:
    """
    Check if code has meaningful verification content:
    - Has requires/ensures clauses
    - Has loop invariants
    - Has proof blocks
    """
    meaningful_patterns = [
        r'\brequires\s*\(',  # requires clauses
        r'\bensures\s*\(',   # ensures clauses
        r'\binvariant\s*\(', # loop invariants
        r'\bproof\s*\{',     # proof blocks
        r'proof\s+fn\s+',    # proof functions
    ]
    
    for pattern in meaningful_patterns:
        if re.search(pattern, code):
            return True
    return False


def compute_quality_score(code: str) -> float:
    """
    Compute a quality/readability score (0.0-1.0) for minimized code.
    Higher score = better quality.
    
    Criteria:
    - Reasonable length (not too short, not too long)
    - Has structure (functions, structs, etc.)
    - Has meaningful content (specs, proofs)
    - Readable formatting
    """
    if not code or len(code.strip()) == 0:
        return 0.0
    
    lines = len(code.splitlines())
    if lines == 0:
        return 0.0
    
    score = 0.0
    
    # Length score (prefer 10-100 lines, penalize extremes)
    if 10 <= lines <= 100:
        score += 0.3
    elif 5 <= lines < 10 or 100 < lines <= 200:
        score += 0.2
    elif lines < 5:
        score += 0.1  # Too short
    else:
        score += 0.15  # Too long
    
    # Structure score (has functions, structs, etc.)
    structure_keywords = ["fn ", "struct ", "enum ", "impl ", "trait "]
    structure_count = sum(code.count(kw) for kw in structure_keywords)
    if structure_count >= 2:
        score += 0.3
    elif structure_count == 1:
        score += 0.2
    else:
        score += 0.1
    
    # Semantic content score
    if is_meaningful_verus_code(code):
        score += 0.3
    else:
        score += 0.1  # Still has some value even without specs
    
    # Readability score (has comments, proper formatting)
    # Check for reasonable whitespace and structure
    non_empty_lines = [l for l in code.splitlines() if l.strip()]
    if len(non_empty_lines) > 0:
        avg_line_length = sum(len(l) for l in non_empty_lines) / len(non_empty_lines)
        if 20 <= avg_line_length <= 100:  # Reasonable line length
            score += 0.1
    
    return min(score, 1.0)  # Cap at 1.0


def meets_dataset_criteria(
    code: str,
    self_contained: bool,
    verifiable: bool,
    quality_score: float
) -> bool:
    """
    Check if code meets ALL criteria for dataset inclusion:
    1. Self-contained: Only std/core/vstd dependencies
    2. Verifiable: Verifies successfully OR fails with recoverable error
    3. Meaningful: Has requires/ensures OR loop invariant OR proof block
    4. Readable: Quality score ≥ 0.5
    5. Properly labeled: Can identify exec/spec/proof regions
    """
    # 1. Self-contained
    if not self_contained:
        return False
    
    # 2. Verifiable (we check this separately, assume True if passed here)
    if not verifiable:
        return False
    
    # 3. Meaningful
    if not is_meaningful_verus_code(code):
        return False
    
    # 4. Readable
    if quality_score < 0.5:
        return False

    return True


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
    if lines == 0 or original_lines == 0:
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


def verus_minimize(code: str, interestingness_test: str, cores: int = 4, timeout: int = 60) -> dict:
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

def attempt_extract(path: Path, repo_url: str, commit_sha: str, repo_root: Optional[Path] = None, interestingness_test: str = "minimal") -> ExtractionResult:
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
            repo_root=repo_root,
            minimized_status=None,
            segments=segments,
        )

    score = score_verus_tokens(text)
    deps = extract_local_dependencies(text)
    # temp_crate = build_temp_crate(text)
    minimized_code = text  # Default to original code
    minimized_status = None
    segments = {}
    minimize_time_ms = None  # Initialize
    quality_score = None
    is_meaningful = False
    meets_criteria = False
    minimum_verifiable = False
    
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
            minimize_time_start = time.perf_counter()
            try:
                minimize_result = verus_minimize(text, interestingness_test)
                minimize_time_ms = int((time.perf_counter() - minimize_time_start) * 1000)
                minimized_code = minimize_result["code"]
                # Map "success" to "succeeded" for consistency with existing code
                minimize_status = minimize_result["status"]
                minimized_status = "succeeded" if minimize_status == "success" else "failed"
                savings = minimize_result.get("savings")
                
                # Verify minimized code still verifies
                minimized_verus_result = eval_verus(minimized_code)
                minimum_verifiable = minimized_verus_result["verus_succeed"]
                
                # Segment the minimized code
                segments = segment_verus_code(minimized_code)
                
                # Compute quality metrics
                quality_score = compute_quality_score(minimized_code)
                is_meaningful = is_meaningful_verus_code(minimized_code)
                meets_criteria = meets_dataset_criteria(
                    minimized_code,
                    self_contained=check_dependencies(deps),
                    verifiable=minimum_verifiable,
                    quality_score=quality_score
                )
            except Exception as e:
                minimize_time_ms = int((time.perf_counter() - minimize_time_start) * 1000)
                minimized_status = "failed"
                minimized_code = text  # Fall back to original
                segments = segment_verus_code(text)
                savings = None
                minimum_verifiable = False
                quality_score = compute_quality_score(text)
                is_meaningful = is_meaningful_verus_code(text)
                meets_criteria = False
            
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
                verus_tokens=score_verus_tokens(minimized_code),
                original_LOC=len(text.splitlines()),
                minimized_LOC=len(minimized_code.splitlines()),
                reduction_ratio=len(minimized_code.splitlines()) / len(text.splitlines()) if len(text.splitlines()) > 0 else None,
                minimum_verifiable=minimum_verifiable,
                self_contained=check_dependencies(deps),
                dependencies=deps,
                verify_time_ms=elapsed_ms,
                repo_url=repo_url,
                commit_sha=commit_sha,
                repo_root=repo_root,
                minimized_status=minimized_status,
                segments=segments,
                minimize_time_ms=minimize_time_ms,
                quality_score=quality_score,
                is_meaningful=is_meaningful,
                meets_criteria=meets_criteria,
            )
        
        # Code doesn't verify - still segment original code
        segments = segment_verus_code(text)
        quality_score = compute_quality_score(text)
        is_meaningful = is_meaningful_verus_code(text)
        meets_criteria = False  # Doesn't verify, so can't meet criteria
        
        return ExtractionResult(
            path,
            "failed",
            "\n".join([str(e) for e in verus_errors]),
            code=text,
            minimized_code=text,
            original_LOC=len(text.splitlines()),
            minimized_LOC=len(text.splitlines()),
            reduction_ratio=1.0,
            minimum_verifiable=False,
            self_contained=check_dependencies(deps),
            dependencies=deps,
            verus_tokens=score_verus_tokens(text),
            verify_time_ms=elapsed_ms,
            repo_url=repo_url,
            commit_sha=commit_sha,
            repo_root=repo_root,
            minimized_status=minimized_status,
            segments=segments,
            minimize_time_ms=None,
            quality_score=quality_score,
            is_meaningful=is_meaningful,
            meets_criteria=meets_criteria,
            verus_errors=verus_errors,
        )
    except subprocess.TimeoutExpired:
        segments = segment_verus_code(text)
        return ExtractionResult(
            path, 
            "timeout", 
            "verification_timeout", 
            code=text,
            minimized_code=text, 
            verus_tokens=score,
            dependencies=deps,
            repo_url=repo_url,
            commit_sha=commit_sha,
            repo_root=repo_root,
            minimized_status=minimized_status,
            segments=segments,
        )
    # finally:
    #     shutil.rmtree(temp_crate, ignore_errors=True)


def print_summary_statistics(results: List[ExtractionResult], out_dir: Path) -> None:
    """Print summary statistics about the extraction results."""
    total = len(results)
    if total == 0:
        summary_text = "\nNo results to summarize.\n"
        print(summary_text)
        summary_file = out_dir / "summary_statistics.txt"
        summary_file.write_text(summary_text, encoding="utf-8")
        return
    
    verified = sum(1 for r in results if r.status == "verified")
    failed = sum(1 for r in results if r.status == "failed")
    skipped = sum(1 for r in results if r.status == "skipped")
    
    # Minimization statistics
    minimized = sum(1 for r in results if r.minimized_status == "succeeded")
    minimize_times = [r.minimize_time_ms for r in results if r.minimize_time_ms is not None]
    avg_minimize_time = sum(minimize_times) / len(minimize_times) if minimize_times else 0
    
    # Quality statistics
    meaningful = sum(1 for r in results if r.is_meaningful)
    meets_criteria_count = sum(1 for r in results if r.meets_criteria)
    quality_scores = [r.quality_score for r in results if r.quality_score is not None]
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
    
    # Self-containment
    self_contained_count = sum(1 for r in results if r.self_contained)
    
    # Reduction statistics
    reductions = [r.reduction_ratio for r in results if r.reduction_ratio is not None and r.reduction_ratio < 1.0]
    avg_reduction = (1 - sum(reductions) / len(reductions)) * 100 if reductions else 0
    
    # Build summary text
    summary_lines = [
        "="*70,
        "EXTRACTION SUMMARY STATISTICS",
        "="*70,
        f"Total files processed:     {total}",
        f"  Verified:                 {verified} ({verified/total*100:.1f}%)",
        f"  Failed:                   {failed} ({failed/total*100:.1f}%)",
        f"  Skipped:                  {skipped} ({skipped/total*100:.1f}%)",
        "",
        "Minimization:",
        f"  Successfully minimized:   {minimized} ({minimized/total*100:.1f}%)",
        f"  Avg minimize time:        {avg_minimize_time/1000:.1f}s per file",
        f"  Avg reduction:           {avg_reduction:.1f}%",
        "",
        "Quality Metrics:",
        f"  Meaningful content:       {meaningful} ({meaningful/total*100:.1f}%)",
        f"  Avg quality score:        {avg_quality:.2f}/1.0",
        f"  Meets dataset criteria:   {meets_criteria_count} ({meets_criteria_count/total*100:.1f}%)",
        "",
        "Self-containment:",
        f"  Self-contained:           {self_contained_count} ({self_contained_count/total*100:.1f}%)",
        "="*70,
    ]
    
    summary_text = "\n".join(summary_lines)
    
    # Write to file
    summary_file = out_dir / "summary_statistics.txt"
    summary_file.write_text(summary_text, encoding="utf-8")
    
    # Also print to console
    print("\n" + summary_text)
    print(f"\nSummary statistics saved to: {summary_file}")


def load_processed_files(manifest_path: Path, repo_root: Path) -> set[Path]:
    """Load set of already processed file paths from existing manifest."""
    processed = set()
    if not manifest_path.exists():
        return processed
    
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    # Extract file_path from metadata.provenance.file_path
                    file_path_str = data.get("metadata", {}).get("provenance", {}).get("file_path", "")
                    if file_path_str:
                        # file_path is now absolute path
                        processed.add(Path(file_path_str).resolve())
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    continue
    except Exception as e:
        print(f"Warning: Could not read existing manifest: {e}")
    
    return processed


def main(repo: Path, out_dir: Path, limit: Optional[int] = None, interestingness_test: str = "minimal", continue_processing: bool = False) -> None:
    # Get git repository information
    repo_url, commit_sha = get_git_info(repo)
    print(f"Repository URL: {repo_url}")
    print(f"Commit SHA: {commit_sha}")
    
    rust_files = find_rust_files(repo)
    print(f"Found {len(rust_files)} Rust files")
    
    # Filter files by line count (50-500 lines)
    rust_files = [f for f in rust_files if 50 <= count_lines(f) <= 500]
    print(f"Filtered to {len(rust_files)} files with 50-500 lines")
    rust_files.sort()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest_3.jsonl"
    
    # Filter out already processed files if continuing
    if continue_processing and manifest.exists():
        processed_files = load_processed_files(manifest, repo)
        original_count = len(rust_files)

        rust_files = [f for f in rust_files if f.resolve() not in processed_files]
        skipped_count = original_count - len(rust_files)
        print(f"Loaded {len(processed_files)} processed files from manifest")
        print(f"Skipping {skipped_count} already processed files, {len(rust_files)}/{original_count} remaining")
    
    results: List[ExtractionResult] = []
    file_mode = "a" if continue_processing and manifest.exists() else "w"

    for idx, path in enumerate(rust_files):
        if limit is not None and idx >= limit:
            break
        result = attempt_extract(path, repo_url=repo_url, commit_sha=commit_sha, repo_root=repo, interestingness_test=interestingness_test)
        results.append(result)
        print(result.to_json())
        # Append each result immediately to avoid losing progress
        with manifest.open(file_mode, encoding="utf-8") as f:
            f.write(result.to_json() + "\n")
        file_mode = "a"  # After first write, always append

    # Print summary statistics
    print_summary_statistics(results, out_dir)




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract dependency-free Verus snippets from a repository")
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Path to the repository root")
    parser.add_argument("--out", type=Path, default=Path("./extracted_snippets"), help="Output directory for manifest")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of files to process")
    parser.add_argument("--interestingness_test", type=str, default="minimal", choices=['minimal', 'invariant', 'spec'], help="Interestingness test for minimization")
    parser.add_argument("--continue", dest="continue_processing", action="store_true", help="Continue processing from existing manifest, skipping already processed files")
    args = parser.parse_args()

    main(args.repo, args.out, args.limit, args.interestingness_test, args.continue_processing)
