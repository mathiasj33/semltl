"""Invoke FishSemML to produce an embedded deterministic Büchi automaton."""

import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path

from jaxltl.ltl.logic.assignment import Assignment


FISHSEMML_JAR = (
    Path(__file__).parents[5]
    / "fishsemml/build/libs/fsemml-dev-all.jar"
)


def _assignment_to_str(assignment: Assignment) -> str:
    return "[" + ",".join(sorted(assignment.true_propositions)) + "]"


def run_fishsemml(
    formula: str,
    propositions: Iterable[str],
    assignments: Iterable[Assignment],
) -> str:
    propositions = tuple(propositions)
    assignments = tuple(assignments)

    with tempfile.NamedTemporaryFile(suffix=".hoa", delete=False) as file:
        output_path = Path(file.name)

    eligible_letters = (
        "[" + ",".join(_assignment_to_str(a) for a in assignments) + "]"
    )

    command = [
        "java",
        "-jar",
        str(FISHSEMML_JAR),
        "embed",
        "--formula",
        formula,
        "--aps",
        ",".join(propositions),
        "--eligible-letters",
        eligible_letters,
        "--output",
        str(output_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            "FishSemML failed.\n"
            f"Command: {' '.join(command)}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    try:
        return output_path.read_text()
    finally:
        output_path.unlink(missing_ok=True)