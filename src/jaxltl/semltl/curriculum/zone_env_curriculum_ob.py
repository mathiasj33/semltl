"""Zones8 curriculum for Boolean combinations of LTLf+ obligations."""

import random
from pathlib import Path

from jaxltl.environments.environment import Environment
from jaxltl.environments.wrappers.wrapper import EnvWrapper
from jaxltl.ltl2action.curriculum.curriculum import (
    Curriculum,
    MultiRandomStage,
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
    ):
        self.sampler = SimpleReachAvoidFormulaSampler(
            depth=depth,
            reach=reach,
            avoid=avoid,
            propositions=propositions,
        )

    def sample(self) -> str:
        formula = self.sampler.sample()
        return f"∃({formula})"


class ObligationGFSampler(Sampler[str]):
    """Sample one recurrence target and optional safety constraints.

    ``GF p`` is represented by the native LTLf+ quantifier ``forall-exists``.
    Only one recurrence target is sampled because conjoining independently
    generated recurrence automata does not preserve Buchi acceptance in the
    current FishSemML product construction.
    """

    def __init__(
        self,
        avoid: int | tuple[int, int],
        propositions: list[str],
    ):
        self.avoid = _as_range(avoid)
        self.propositions = propositions

    def sample(self) -> str:
        avoid_count = random.randint(*self.avoid)
        reach = random.choice(self.propositions)
        remaining = [p for p in self.propositions if p != reach]
        avoid = random.sample(remaining, min(avoid_count, len(remaining)))

        obligations = [f"∀∃({reach})"]
        if avoid:
            obligations.append(f"∀(!({' | '.join(avoid)}))")
        return " & ".join(obligations)


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
        return f"∃({finite_formula})"


def _as_range(value: int | tuple[int, int]) -> tuple[int, int]:
    return (value, value) if isinstance(value, int) else value


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
            # 6. Final mixture of existential and universal obligations
            MultiRandomStage(
                [
                    RandomCurriculumStage(
                        sampler=ObligationReachAvoidSampler(
                            depth=(1, 2),
                            reach=(1, 2),
                            avoid=(0, 2),
                            propositions=propositions,
                        ),
                        threshold=None,
                    ),
                    RandomCurriculumStage(
                        sampler=ObligationGFSampler(
                            avoid=(0, 2),
                            propositions=propositions,
                        ),
                        threshold=None,
                    ),
                ],
                probs=[0.7, 0.3],
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
    """Create seven single-formula stages for validating LTLf+ integration."""

    propositions = list(env.propositions)
    required_propositions = 2
    if len(propositions) < required_propositions:
        raise ValueError(
            "The validation curriculum requires at least two propositions."
        )
    p, q = propositions[:required_propositions]
    formulas = [
        f"∃(F {p})",
        f"∃(F({p} & F {q}))",
        f"∃((!{q}) U {p})",
        f"∃({p} & N({q}))",
        f"∀({p} -> N({q}))",
        f"∀∃({p})",
        f"∀∃({p}) & ∀(!{q})",
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
