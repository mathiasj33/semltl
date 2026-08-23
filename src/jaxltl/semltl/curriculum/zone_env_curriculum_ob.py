"""Zones8 curriculum for Boolean combinations of LTLf+ obligations."""

import random
from pathlib import Path

from jaxltl.environments.environment import Environment
from jaxltl.environments.wrappers.wrapper import EnvWrapper
from jaxltl.ltl2action.curriculum.curriculum import (
    Curriculum,
    RandomCurriculumStage,
    Sampler,
)
from jaxltl.ltl2action.curriculum.simple_samplers import (
    SimpleReachAvoidFormulaSampler,
)
from jaxltl.semltl.curriculum.batching import SemanticLDBABatcher


class FixedFormulaSampler(Sampler[str]):
    """Return one fixed formula for a diagnostic curriculum stage."""

    def __init__(self, formula: str):
        self.formula = formula

    def sample(self) -> str:
        return self.formula


class ObligationReachAvoidSampler(Sampler[str]):
    """Convert a sampled co-safety task into an existential LTLf+ obligation.

    The temporal operators in the sampled finite-trace formula must be retained:
    for example, ``exists(F p)`` permits reaching ``p`` later, whereas
    ``exists(p)`` requires ``p`` in the first automaton letter.
    """

    def __init__(
        self,
        depth: int | tuple[int, int],
        reach: int | tuple[int, int],
        avoid: int | tuple[int, int],
        propositions: list[str],
        remove_outer_finally: bool = False,
        shift_to_next: bool = False,
        simplify_terminal_true: bool = False,
    ):
        self.sampler = SimpleReachAvoidFormulaSampler(
            depth=depth,
            reach=reach,
            avoid=avoid,
            propositions=propositions,
        )
        self.remove_outer_finally = remove_outer_finally
        self.shift_to_next = shift_to_next
        self.simplify_terminal_true = simplify_terminal_true

    def sample(self) -> str:
        formula = self.sampler.sample()
        if self.remove_outer_finally:
            formula = _remove_outer_finally(formula)
        if self.simplify_terminal_true:
            formula = _simplify_terminal_true(formula)
        if self.shift_to_next:
            formula = f"X({formula})"
        return f"∃({formula})"


class ObligationWeakNextSampler(Sampler[str]):
    """Sample LTLf+ obligations containing the weak-next operator ``N``.

    Existential samples describe short finite sequences. Universal samples
    describe next-step response rules. Both forms keep ``N`` inside the finite
    formula governed by the outer LTLf+ quantifier.
    """

    def __init__(
        self,
        depth: int | tuple[int, int],
        propositions: list[str],
        universal_probability: float = 0.5,
    ):
        self.depth = _as_range(depth)
        self.propositions = propositions
        self.universal_probability = universal_probability

    def sample(self) -> str:
        depth = min(random.randint(*self.depth), len(self.propositions))
        selected = random.sample(self.propositions, depth)

        if random.random() < self.universal_probability:
            antecedent, consequent = selected[0], selected[1]
            return f"∀({antecedent} -> N({consequent}))"

        finite_formula = selected[-1]
        for proposition in reversed(selected[:-1]):
            finite_formula = f"{proposition} & N({finite_formula})"
        return f"∃(F({finite_formula}))"


def _as_range(value: int | tuple[int, int]) -> tuple[int, int]:
    return (value, value) if isinstance(value, int) else value


def _remove_outer_finally(formula: str) -> str:
    """Remove the outer ``F(...)`` from a simple reach formula."""

    if formula.startswith("F(") and formula.endswith(")"):
        return formula[2:-1]
    raise ValueError(f"Expected a formula with an outer F, got: {formula}")


def _simplify_terminal_true(formula: str) -> str:
    """Simplify the ``(p) & true`` suffix of a one-step reach formula."""

    suffix = " & true"
    if not formula.endswith(suffix):
        raise ValueError(f"Expected a formula ending in '{suffix}', got: {formula}")
    formula = formula[: -len(suffix)]
    if formula.startswith("(") and formula.endswith(")"):
        return formula[1:-1]
    return formula


def make(env: Environment | EnvWrapper, load_path: Path | None = None) -> Curriculum:
    """Create the Zones8 curriculum using only LTLf+ obligation formulas."""

    propositions = list(env.propositions)
    return Curriculum(
        [
            # 1. Simple existential reach tasks
            RandomCurriculumStage(
                sampler=ObligationReachAvoidSampler(
                    depth=1,
                    reach=1,
                    avoid=0,
                    propositions=propositions,
                ),
                threshold=0.9,
            ),
            # 2. Existential sequential reach tasks of depth 2
            RandomCurriculumStage(
                sampler=ObligationReachAvoidSampler(
                    depth=2,
                    reach=1,
                    avoid=0,
                    propositions=propositions,
                    remove_outer_finally=True,
                    shift_to_next=True,
                ),
                threshold=0.95,
            ),
            # 3. Simple existential reach-avoid tasks
            RandomCurriculumStage(
                sampler=ObligationReachAvoidSampler(
                    depth=1,
                    reach=1,
                    avoid=1,
                    propositions=propositions,
                ),
                threshold=0.95,
            ),
            # 4. Existential reach-avoid tasks of depth 2
            RandomCurriculumStage(
                sampler=ObligationReachAvoidSampler(
                    depth=2,
                    reach=1,
                    avoid=1,
                    propositions=propositions,
                ),
                threshold=0.9,
            ),
            # 5. LTLf+ weak-next sequence and response obligations
            RandomCurriculumStage(
                sampler=ObligationWeakNextSampler(
                    depth=(2, 3),
                    propositions=propositions,
                ),
                threshold=0.9,
            ),
            # 6. General existential reach and reach-avoid obligations
            RandomCurriculumStage(
                sampler=ObligationReachAvoidSampler(
                    depth=(1, 2),
                    reach=(1, 2),
                    avoid=(0, 2),
                    propositions=propositions,
                ),
                threshold=None,
            ),
        ],
        num_samples=10_000,
        batcher=SemanticLDBABatcher(),
        env=env,
        load_path=load_path,
    )


def make_validation(
    env: Environment | EnvWrapper, load_path: Path | None = None
) -> Curriculum:
    """Create fixed examples of every formula family in the main curriculum."""

    propositions = list(env.propositions)
    required_propositions = 4
    if len(propositions) < required_propositions:
        raise ValueError(
            "The validation curriculum requires at least two propositions."
        )
    p, q, r, s = propositions[:required_propositions]
    formulas = [
        f"∃( {p})",
        f"∃( ({p} & F {q}))",
        f"∀((!{q}) U {p})",
        f"∃((!{q}) U ({p} & ((!{r}) U {s})))",
        f"∃(F({p} & N({q})))",
        f"∀({p} -> N({q}))",
        f"∃(!({q} | {r}) U ({p} | {s}))",
    ]
    stages = [
        RandomCurriculumStage(
            sampler=FixedFormulaSampler(formula),
            threshold=0.9 if index < len(formulas) - 1 else None,
        )
        for index, formula in enumerate(formulas)
    ]
    return Curriculum(
        stages,
        num_samples=1,
        batcher=SemanticLDBABatcher(),
        env=env,
        load_path=load_path,
    )
