import os
import logging
import json
import subprocess
import tempfile
from enum import Enum
from typing import Tuple, List, Optional, Dict, Any
import re


class Verus:
    def __init__(self):
        self.verus_path = None

    def set_verus_path(self, path):
        self.verus_path = os.path.realpath(path)
        self.vstd_path = os.path.realpath(os.path.join(self.verus_path, "../../../vstd/"))
        # print(f"verus path: {self.verus_path}")
        # print(f"vstd path: {self.vstd_path}")


verus = Verus()
verus.set_verus_path("/Users/ameliakuang/Repos/verus/verus")

class VerusErrorType(Enum):
    PreCondFail = 1
    PostCondFail = 2
    InvFailEnd = 3
    InvFailFront = 4
    DecFailEnd = 5
    DecFailCont = 6
    TestAssertFail = 7
    RecommendNotMet = 8
    AssertFail = 11
    ArithmeticFlow = 12
    MismatchedType = 13
    PreCondFailVecLen = 14
    MissImpl = 15
    Other = 16
    ensure_private = 17
    require_private = 18
    MissingImport = 19
    TypeAnnotation = 20
    ConstructorFailTypeInvariant = 21
    CannotCallFunc = 22
    RequiresOldSelf = 23
    PubSpecVisibility = 24


m2VerusError = {
    "precondition not satisfied": VerusErrorType.PreCondFail,
    "postcondition not satisfied": VerusErrorType.PostCondFail,
    "invariant not satisfied at end of loop body": VerusErrorType.InvFailEnd,
    "invariant not satisfied before loop": VerusErrorType.InvFailFront,
    "decreases not satisfied at end of loop": VerusErrorType.DecFailEnd,
    "decreases not satisfied at continue": VerusErrorType.DecFailCont,
    "recommendation not met": VerusErrorType.RecommendNotMet,
    "assertion failed": VerusErrorType.AssertFail,
    "possible arithmetic underflow/overflow": VerusErrorType.ArithmeticFlow,
    "mismatched types": VerusErrorType.MismatchedType,
    "in 'ensures' clause of public function, cannot access any field of a datatype where one or more fields are private": VerusErrorType.ensure_private,
    "in 'requires' clause of public function, cannot refer to private function": VerusErrorType.require_private,
    "cannot find macro `verus` in this scope": VerusErrorType.MissingImport,
    "type annotations needed": VerusErrorType.TypeAnnotation,
    "constructed value may fail to meet its declared type invariant": VerusErrorType.ConstructorFailTypeInvariant,
    "cannot call function": VerusErrorType.CannotCallFunc,
    "in requires, use `old(self)` to refer to the pre-state of an &mut variable": VerusErrorType.RequiresOldSelf,
    "non-private spec function must be marked open or closed": VerusErrorType.PubSpecVisibility,
}

VerusError2m = {v: k for k, v in m2VerusError.items()}


class VerusErrorLabel(Enum):
    NullLabel = 0
    FailedThisPostCond = 1
    FailedThisPreCond = 2
    RecmdNotMet = 3
    EndOfFunc = 4


m2VerusErrorLabel = {
    None: VerusErrorLabel.NullLabel,
    "failed this postcondition": VerusErrorLabel.FailedThisPostCond,
    "failed precondition": VerusErrorLabel.FailedThisPreCond,
    "recommendation not met": VerusErrorLabel.RecmdNotMet,
    "at the end of the function body": VerusErrorLabel.EndOfFunc,
}

VerusErrorLabel2m = {v: k for k, v in m2VerusErrorLabel.items()}



class ErrorText:
    def __init__(self, text):
        self.text = text["text"]
        self.hl_start = text["highlight_start"]
        self.hl_end = text["highlight_end"]

class ErrorTrace:
    def __init__(self, span):
        self.fname = span["file_name"]
        self.lines = (int(span["line_start"]), int(span["line_end"]))
        if span["label"] not in m2VerusErrorLabel:
            self.label = VerusErrorLabel.NullLabel
        else:
            self.label = m2VerusErrorLabel[span["label"]]
        self.text = [ErrorText(t) for t in span["text"]]
        self.vstd_err = self.fname.startswith(os.path.realpath(verus.vstd_path))
        self.strlabel = span["label"]

    def is_vstd_err(self):
        return self.vstd_err

    def get_text(self, snippet=True, pre=4, post=2):
        ret = f"{VerusErrorLabel2m[self.label]}\n" if VerusErrorLabel2m[self.label] else ""
        if not snippet or len(self.text) <= pre + post + 1:
            return ret + "\n".join([t.text for t in self.text])
        else:
            return ret + "\n".join(
                [t.text for t in self.text[:pre]] + ["..."] + [t.text for t in self.text[-post:]]
            )

    # TO be refined
    def get_highlights(self):
        return [t.text[t.hl_start - 1 : t.hl_end - 1] for t in self.text]

    def get_lines(self):
        return self.lines

class VerusError:
    def __init__(self, err: dict, code: str = None):
        # Store the raw message text and spans
        self.error_text = err["message"]
        self.spans = err["spans"] if "spans" in err else []
        self.logger = logging.getLogger("VerusError")
        self.code = code  # Store code for test function detection

        # Create the trace first so we can use it for error classification
        self.trace = [ErrorTrace(t) for t in self.spans]  # Bottom-up stack trace

        # Get the full error message including span labels
        if self.spans:
            span_labels = [span.get("label", "") for span in self.spans if "label" in span]
            self.error_text = (
                f"{self.error_text} ({'; '.join(label for label in span_labels if label)})"
            )

        # Default to 'Other' unless a partial match is found
        self.error = VerusErrorType.Other

        # Try to match by substring against known keys
        for known_msg, err_type in m2VerusError.items():
            if known_msg in self.error_text:
                # Special case: don't treat empty function body errors as type errors
                if err_type == VerusErrorType.MismatchedType:
                    if "implicitly returns `()`" in self.error_text:
                        continue
                self.error = err_type
                break

        # Handle any special-cases not captured in the dictionary
        if self.error == VerusErrorType.Other:
            if "not all trait items implemented, missing" in self.error_text:
                self.error = VerusErrorType.MissImpl

        # Special case: TestAssertFail is an assertion failure inside a test function
        # Use code + line numbers to find containing function
        if self.error == VerusErrorType.AssertFail and self.code and self.trace:
            code_lines = self.code.split("\n")

            for trace in self.trace:
                error_line = trace.lines[0]  # 1-indexed

                # Search backwards from error line to find function definition
                for i in range(error_line - 1, max(0, error_line - 51), -1):
                    if i < len(code_lines):
                        line = code_lines[i]
                        # Match function definition (with optional attributes like #[verifier::loop_isolation])
                        fn_match = re.search(r"^\s*(?:#\[.*?\]\s*)?fn\s+(\w+)\s*\(", line)
                        if fn_match:
                            func_name = fn_match.group(1)
                            # Check if function name contains "test"
                            if "test" in func_name.lower():
                                self.logger.debug(
                                    f"Detected test assertion failure in function '{func_name}' at line {error_line}"
                                )
                                self.error = VerusErrorType.TestAssertFail
                                break
                            else:
                                # Found a non-test function, stop searching
                                self.logger.debug(
                                    f"Assertion in non-test function '{func_name}' at line {error_line}"
                                )
                                break

                if self.error == VerusErrorType.TestAssertFail:
                    break
        elif self.error == VerusErrorType.AssertFail:
            # Debug: log why test detection didn't run
            if not self.code:
                self.logger.debug(f"Test assertion detection skipped: code is empty or None")
            elif not self.trace:
                self.logger.debug(f"Test assertion detection skipped: trace is empty")

        # a subtype of precondfail that often requires separate treatment
        if self.error == VerusErrorType.PreCondFail:
            if self.trace and "i < vec.view().len()" in self.trace[0].get_text():
                self.error = VerusErrorType.PreCondFailVecLen

    def __str__(self):
        return f"{self.error}: {self.error_text}"

    def get_miss_impl_funcs(self):
        if self.error != VerusErrorType.MissImpl:
            return []

        def extract_function_names(text):
            pattern = r"`(\w+)`"
            matches = re.findall(pattern, text)
            return matches

        function_names = extract_function_names(self.error_text)
        return function_names

    def get_text(self, snippet=True, pre=4, post=2, topdown=True):
        traces = []
        for t in self.trace:
            t_text = t.get_text(snippet, pre, post)
            if t_text and t_text not in traces:
                traces.append(t_text)

        if topdown:
            traces = traces[::-1]

        span_texts = []
        for span in self.spans:
            if "text" in span:
                highlights = []
                for t in span["text"]:
                    text = t["text"][t["highlight_start"] - 1 : t["highlight_end"] - 1]
                    highlights.append(text)
                highlight_text = " ".join(highlights)
                label = span["label"]
                span_texts += [f"{label}: {highlight_text}"]
        return "\n".join(traces) + "\n  " + "\n  ".join(span_texts)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, VerusError):
            return False

        return self.error_text == value.error_text and self.get_text() == value.get_text()



def verus_succeed(verus_result) -> bool:
    if not verus_result:
        Exception("No Verus result")
    return verus_result["verification-results"]["success"]


# Run verus on the code and parse the output.
def eval_verus(
    code: str,
    max_errs=5,
    json_mode=True,
    func_name=None,
    no_verify=False,
    log_dir=None,
    expand_errors=False,
) -> Dict[str, Any]:

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write(code)
        code_path = f.name
    multiple_errors = f"--multiple-errors {max_errs}" if max_errs > 0 else ""
    err_format = "--output-json --error-format=json" if json_mode else ""
    # cmd = (f"{self.verus_path} {multiple_errors} {err_format} {code_path}").split(" ")
    # Bug fix: code_path may contain white space
    # Use --crate-type=lib like verify_snippet does to avoid requiring main() function
    cmd = (f"verus --crate-type=lib {multiple_errors} {err_format}").split(" ")
    cmd += [code_path]
    if func_name:
        cmd += ["--verify-function", func_name, "--verify-root"]
    if no_verify:
        cmd += ["--no-verify"]
    if not (log_dir is None):
        # Add log to the default file log_dir if log_dir is not empty.
        # When this is enabled, verus will produce log, including:
        # - callgraph,
        # - verus intermediate language (vir),
        # - and smt file
        # Maybe useful for in-depth analysis
        if log_dir != "":
            cmd += ["--log-dir", log_dir]
        cmd += ["--log-all"]
    if expand_errors:
        # When expand_errors = true,
        # verus will report which postcond is established and which are not
        cmd += ["--expand-errors"]
    # self.logger.info(f"Running command: {' '.join(cmd)}")
    m = subprocess.run(cmd, capture_output=True, text=True)
    verus_out = m.stdout
    rustc_out = m.stderr
    os.unlink(code_path)

    if not json_mode:
        return {
            "verus_succeed": False,
            "verified_count": 0,
            "error_count": 0,
            "verus_errors": [],
            "compilation_error": False,
        }

    try:
        verus_result = json.loads(verus_out)
    except json.JSONDecodeError as e:
        verus_result = None

    # Extract verification results from verus_result JSON
    verus_succeeded = False
    verified_count = 0
    error_count = 0
    
    if verus_result and "verification-results" in verus_result:
        vr = verus_result["verification-results"]
        verus_succeeded = vr.get("success", False) if isinstance(vr, dict) else False
        verified_count = vr.get("verified", 0) if isinstance(vr, dict) else 0
        error_count = vr.get("errors", 0) if isinstance(vr, dict) else 0

    # Initialize compilation_error
    compilation_error = False
    # If verus succeed, but rustc failed, then it is a compilation error.
    if verus_result and verus_succeed(verus_result) and m.returncode != 0:
        compilation_error = True
    verus_errors = []
    rustc_result = []
    for rust_err in rustc_out.split("\n")[:-1]:
        try:
            e = json.loads(rust_err)
        except json.JSONDecodeError as e:
            continue
        if not isinstance(e, dict):
            print(f"Unexpected rust err output: {e}")
            continue
        rustc_result.append(e)
        if "level" in e and e["level"] == "error":
            if "message" in e and "aborting due to" in e["message"].lower():
                continue  # Skip trivial aborting errors.
            # Make unclosed delimiter error worse than other errors
            if "unclosed delimiter" in e["message"]:
                verus_errors.append(
                    VerusError(
                        {"message": "unclosed delimiter", "spans": []},
                        code=code,
                    )
                )
            verus_errors.append(VerusError(e, code=code))
    
    # Return dictionary with verification results
    return {
        "verus_succeed": verus_succeeded,
        "verified_count": verified_count,
        "error_count": error_count,
        "verus_errors": verus_errors,
        "compilation_error": compilation_error,
    }