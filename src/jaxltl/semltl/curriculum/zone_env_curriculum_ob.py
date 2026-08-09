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


class ObligationReachAvoidSampler(Sampler[str]):
    """Convert a sampled co-safety task into an existential LTLf+ obligation.

    For pure reach tasks, the sampler returns an LTL formula with an outer ``F``.
    Since that outer eventuality is represented by ``exists`` in LTLf+, it is
    removed before the remaining finite-trace formula is quantified. Reach-avoid
    tasks start with ``U`` and are quantified without further rewriting.
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
        finite_formula = _remove_outer_finally(formula)
        return f"∃({finite_formula})"


class ObligationGFSampler(Sampler[str]):
    """Sample recurrence and safety tasks as universal obligations.

    ``GF p`` is represented as ``forall(F p)`` and ``G(!q)`` as
    ``forall(!q)``.
    """

    def __init__(
        self,
        reach: int | tuple[int, int],
        avoid: int | tuple[int, int],
        propositions: list[str],
    ):
        self.reach = _as_range(reach)
        self.avoid = _as_range(avoid)
        self.propositions = propositions

    def sample(self) -> str:
        reach_count = random.randint(*self.reach)
        avoid_count = random.randint(*self.avoid)
        reach = random.sample(self.propositions, reach_count)
        remaining = [p for p in self.propositions if p not in reach]
        avoid = random.sample(remaining, min(avoid_count, len(remaining)))

        obligations = [f"∀(F {proposition})" for proposition in reach]
        if avoid:
            obligations.append(f"∀(!({' | '.join(avoid)}))")
        return " & ".join(obligations)


class ObligationFGSampler(Sampler[str]):
    """Sample persistence and safety tasks as LTLf+ obligations.

    ``FG p`` is represented as ``exists(G p)`` and ``G(!q)`` as
    ``forall(!q)``.
    """

    def __init__(
        self,
        avoid: int | tuple[int, int],
        propositions: list[str],
    ):
        self.avoid = _as_range(avoid)
        self.propositions = propositions

    def sample(self) -> str:
        persistent = random.choice(self.propositions)
        avoid_count = random.randint(*self.avoid)
        remaining = [p for p in self.propositions if p != persistent]
        avoid = random.sample(remaining, min(avoid_count, len(remaining)))

        obligations = [f"∃(G {persistent})"]
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


def _remove_outer_finally(formula: str) -> str:
    """Remove the outer ``F(...)`` emitted for a pure reach task."""

    if formula.startswith("F(") and formula.endswith(")"):
        return formula[2:-1]
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
                            reach=(2, 3),
                            avoid=(0, 2),
                            propositions=propositions,
                        ),
                        threshold=None,
                    ),
                    RandomCurriculumStage(
                        sampler=ObligationFGSampler(
                            avoid=(0, 2),
                            propositions=propositions,
                        ),
                        threshold=None,
                    ),
                ],
                probs=[0.5, 0.25, 0.25],
                threshold=None,
            ),
        ],
        num_samples=10_000,
        batcher=SemanticLDBABatcher(),
        env=env,
        load_path=load_path,
    )
