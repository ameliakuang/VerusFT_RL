"""
Analysis script to read JSONL manifest files and generate a CSV summary.
"""
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any


def read_jsonl_files(directories: List[Path]) -> List[Dict[str, Any]]:
    """Read all manifest_*.jsonl files from the given directories."""
    all_entries = []
    
    for directory in directories:
        if not directory.exists():
            continue
        
        for manifest_file in directory.glob("manifest_*.jsonl"):
            print(f"Reading {manifest_file}")
            try:
                with manifest_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                all_entries.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
            except Exception as e:
                print(f"Error reading {manifest_file}: {e}")
    
    return all_entries


def extract_fields(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the required fields from a JSON entry."""
    m = entry.get("metadata", {})
    prov = m.get("provenance", {})
    verif = m.get("verification", {})
    qual = m.get("quality", {})
    labeling = m.get("labeling", {})
    
    min_time_ms = verif.get("minimize_time_ms")
    min_status = verif.get("minimized_status")
    verif_status = verif.get("status")
    error_type = verif.get("error_type")
    has_invariant = labeling.get("has_invariant")
    has_proof_block = labeling.get("has_proof_block")
    requires_count = labeling.get("requires_count")
    ensures_count = labeling.get("ensures_count")
    
    return {
        "file_path": prov.get("file_path", ""),
        "original_LOC": qual.get("original_LOC"),
        "minimized_LOC": qual.get("minimized_LOC"),
        "reduction_ratio": qual.get("reduction_ratio"),
        "minimization_time": min_time_ms / 1000.0 if min_time_ms else None,
        "minimization_success": min_status == "succeeded" if min_status else None,
        "readability_score": qual.get("readability_score"),
        "self_contained": qual.get("self_contained"),
        "dependencies": qual.get("dependencies"),
        "verifiable": verif_status == "verified" if verif_status else None,
        "error_type": error_type,
        "has_invariant": has_invariant,
        "has_proof_block": has_proof_block,
        "requires_count": requires_count,
        "ensures_count": ensures_count,
        "original_code": entry.get("original_code", ""),
        "minimized_code": entry.get("minimized_code", ""),
        "all_snapshots": entry.get("all_snapshots", [])
    }


def generate_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    """Generate a CSV file from the extracted rows."""
    if not rows:
        print("No entries to write to CSV")
        return
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"CSV written to {output_path} ({len(rows)} entries)")


def print_stats(rows: List[Dict[str, Any]], print_anomalies: bool = False) -> None:
    """Print summary statistics."""
    if not rows:
        return
    
    df = pd.DataFrame(rows)
    
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"Total entries: {len(df)}")
    
    # Count non-None values
    for field in ["original_LOC", "minimized_LOC", "reduction_ratio", "minimization_time", 
                  "minimization_success", "readability_score", "self_contained", "verifiable"]:
        count = df[field].notna().sum()
        print(f"  With {field} ！= None: {count}")
    
    # Compute DataFrame summary statistics for numeric metrics
    numeric_fields = {
        "minimization_time": "Minimization Time (seconds)",
        "reduction_ratio": "Reduction Ratio",
        "readability_score": "Readability Score",
        "original_LOC": "original line of code"
    }
    
    for field, label in numeric_fields.items():
        if field in df.columns:
            series = df[field].dropna()
            if len(series) > 0:
                print(f"\n{label} Statistics:")
                print(f"  Count:    {len(series)}")
                print(f"  Mean:     {series.mean():.3f}")
                print(f"  Median:   {series.median():.3f}")
                print(f"  Std Dev:  {series.std():.3f}")
                print(f"  Min:      {series.min():.3f}")
                print(f"  Max:      {series.max():.3f}")
                print(f"  25th %ile: {series.quantile(0.25):.3f}")
                print(f"  75th %ile: {series.quantile(0.75):.3f}")
    
    # Calculate percentages for verified and minimized
    print("\n" + "="*70)
    print("VERIFICATION & MINIMIZATION RATES")
    print("="*70)
    
    # Percentage verified
    if "verifiable" in df.columns:
        verifiable_total = df["verifiable"].notna().sum()
        verifiable_count = (df["verifiable"] == True).sum()
        if verifiable_total > 0:
            verifiable_pct = (verifiable_count / verifiable_total) * 100
            print(f"Verified: {verifiable_count}/{verifiable_total} ({verifiable_pct:.1f}%)")
            
            # Show error types for failed verifications
            failed = df[df["verifiable"] == False]
            if len(failed) > 0 and "error_type" in df.columns:
                print(f"\nFailed Verification Error Types ({len(failed)} entries):")
                error_counts = failed["error_type"].value_counts()
                for error_type, count in error_counts.items():
                    if pd.notna(error_type):
                        pct = (count / len(failed)) * 100
                        print(f"  {error_type}: {count} ({pct:.1f}%)")
    
    # Percentage minimized (minimization_success == True)
    if "minimization_success" in df.columns:
        minimized_total = df["minimization_success"].notna().sum()
        minimized_count = (df["minimization_success"] == True).sum()
        if minimized_total > 0:
            minimized_pct = (minimized_count / minimized_total) * 100
            print(f"Minimized: {minimized_count}/{minimized_total} ({minimized_pct:.1f}%)")
    
    # Other rates
    fields = {
        "self_contained": "Self-contained",
        "has_invariant": "Has Invariant in Minimized Code",
        "has_proof_block": "Has Proof Block in Minimized Code",
        "verifiable": "Verifiable",
        "requires_count": "Has Requires in Minimized Code",
        "ensures_count": "Has Ensures in Minimized Code",
    }
    
    for field, label in fields.items():
        if field in df.columns:
            total = len(df)
            # Boolean fields: count True; Integer fields: count > 0
            if field in ["requires_count", "ensures_count"]:
                count = (df[field].notna() & (df[field] > 0)).sum()
            else:
                count = (df[field] == True).sum()
            if total > 0:
                pct = (count / total) * 100
                print(f"{label}: {count}/{total} ({pct:.1f}%)")
    
    # Check for anomalous reduction ratios (> 1 means minimized code is larger than original)
    if print_anomalies and "reduction_ratio" in df.columns:
        anomalous = df[df["reduction_ratio"].notna() & (df["reduction_ratio"] > 1.0)]
        if len(anomalous) > 0:
            print("\n" + "="*70)
            print(f"WARNING: Found {len(anomalous)} entries with reduction_ratio > 1.0")
            print("="*70)
            print("These entries have minimized code larger than original (unexpected):")
            print()
            for idx, row in anomalous.iterrows():
                print(f"File: {row.get('file_path', 'N/A')}")
                print(f"  Original LOC: {row.get('original_LOC', 'N/A')}")
                print(f"  Minimized LOC: {row.get('minimized_LOC', 'N/A')}")
                print(f"  Reduction Ratio: {row.get('reduction_ratio', 'N/A'):.3f}")
                print(f"  Minimization Success: {row.get('minimization_success', 'N/A')}")
                print(f"########################################################")
                print(f"  Code: {row.get('original_code', 'N/A')}")
                print(f"########################################################")
                print(f"  Minimized Code: {row.get('minimized_code', 'N/A')}")
                print(f"########################################################")
                print()
                break


def main():
    """Main function to run the analysis."""
    base_dir = Path("/Users/ameliakuang/Repos/verus-sft/VerusFT_RL/extracted_snippets")
    directories = [
        base_dir / "autoverus_examples",
        base_dir / "vericoding_benchmarks_verus",
        base_dir / "verus_examples",
    ]
    
    print("Reading JSONL files...")
    entries = read_jsonl_files(directories)
    print(f"Total entries read: {len(entries)}")
    
    rows = [extract_fields(entry) for entry in entries]
    
    output_path = base_dir / "analysis_summary.csv"
    print("\nGenerating CSV...")
    generate_csv(rows, output_path)
    
    print_stats(rows)


if __name__ == "__main__":
    main()
