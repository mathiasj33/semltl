"""Utility functions for calling SemML to convert LTL formulas to LDBAs."""

import subprocess
from collections.abc import Iterable

from jaxltl import DATA_DIR, DEPENDENCIES_DIR
from jaxltl.ltl.logic.assignment import Assignment

SEMML_PATH = DEPENDENCIES_DIR / "semml/scripts_semml/embedd_ldba.py"


def run_semml(
    formula: str,
    propositions: Iterable[str],
    assignments: Iterable[Assignment],
    use_attention: bool = True,
) -> str:
    """Convert an LTL formula to a LDBA using SemML."""

    output_path = DATA_DIR / "tmp" / "semml_ldba.hoa"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    assignments_str = f"[{','.join([_assignment_to_str(a) for a in assignments])}]"
    command = [
        "python3",
        SEMML_PATH.as_posix(),
        "--formula",
        formula,
        "--aps",
        f"{','.join(propositions)}",
        "--eligibleLetters",
        assignments_str,
        "--outputPath",
        output_path.as_posix(),
    ]
    if not use_attention:
        command += ["--featureList", "260112"]
    run = subprocess.run(command, capture_output=True, text=True, check=False)
    if run.stderr != "":
        raise RuntimeError(
            f"SemML call `{' '.join(command)}` resulted in an error.\nError: {run.stderr}."
        )
    if not output_path.exists():
        raise RuntimeError(
            "SemML did not produce an output file.\nSemML output:\n" + run.stdout
        )
    with open(output_path) as f:
        ldba = f.read()
    output_path.unlink()
    return ldba


def _assignment_to_str(assignment: Assignment) -> str:
    return "[" + ",".join(sorted(assignment.true_propositions)) + "]"
