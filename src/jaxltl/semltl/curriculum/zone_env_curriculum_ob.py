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
        quantifier: str = "E",
    ):
        self.sampler = SimpleReachAvoidFormulaSampler(
            depth=depth,
            reach=reach,
            avoid=avoid,
            propositions=propositions,
        )
        self.quantifier = quantifier

    def sample(self) -> str:
        formula = self.sampler.sample()
        return f"{self.quantifier}({formula})"


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
            return f"A({antecedent} -> N({consequent}))"

        finite_formula = selected[-1]
        for proposition in reversed(selected[:-1]):
            finite_formula = f"{proposition} & N({finite_formula})"
        return f"E(F({finite_formula}))"


class SafetyGuaranteeObligationSampler(Sampler[str]):
    """Sample conjunctions of existential guarantees and universal safety."""

    def __init__(
        self,
        guarantees: int,
        avoid: int,
        propositions: list[str],
    ):
        if guarantees < 1 or avoid < 1:
            raise ValueError("At least one guarantee and one avoidance are required.")
        if guarantees + avoid > len(propositions):
            raise ValueError("Not enough propositions for distinct task clauses.")
        self.guarantees = guarantees
        self.avoid = avoid
        self.propositions = propositions

    def sample(self) -> str:
        selected = random.sample(
            self.propositions, self.guarantees + self.avoid
        )
        guarantee_propositions = selected[: self.guarantees]
        avoid_propositions = selected[self.guarantees :]

        guarantee = " & ".join(
            f"F({proposition})" for proposition in guarantee_propositions
        )
        if len(avoid_propositions) == 1:
            safety = f"!{avoid_propositions[0]}"
        else:
            safety = f"!({' | '.join(avoid_propositions)})"
        return f"E({guarantee}) & A(G({safety}))"


def _as_range(value: int | tuple[int, int]) -> tuple[int, int]:
    return (value, value) if isinstance(value, int) else value


def make(env: Environment | EnvWrapper, load_path: Path | None = None) -> Curriculum:
    """Create a cumulative Zones8 curriculum of LTLf+ obligations.

    Each stage focuses on one new formula family while replaying samples from
    earlier stages. This limits catastrophic forgetting as the curriculum
    advances and makes the final stage representative of the complete task
    distribution rather than only its most difficult family.
    """

    propositions = list(env.propositions)

    reach = ObligationReachAvoidSampler(
        depth=1,
        reach=1,
        avoid=0,
        propositions=propositions,
    )
    sequential_reach = ObligationReachAvoidSampler(
        depth=2,
        reach=1,
        avoid=0,
        propositions=propositions,
    )
    reach_avoid = ObligationReachAvoidSampler(
        depth=1,
        reach=1,
        avoid=1,
        propositions=propositions,
        quantifier="E",
    )
    sequential_reach_avoid = ObligationReachAvoidSampler(
        depth=2,
        reach=1,
        avoid=1,
        propositions=propositions,
    )
    weak_next = ObligationWeakNextSampler(
        depth=2,
        propositions=propositions,
        universal_probability=0.0,
    )
    safety_guarantee = SafetyGuaranteeObligationSampler(
        guarantees=1,
        avoid=1,
        propositions=propositions,
    )
    multi_safety_guarantee = SafetyGuaranteeObligationSampler(
        guarantees=2,
        avoid=2,
        propositions=propositions,
    )

    def random_stage(sampler: Sampler[str]) -> RandomCurriculumStage[str]:
        """Wrap a sampler for use inside a mixed curriculum stage."""

        return RandomCurriculumStage(sampler=sampler, threshold=None)

    return Curriculum(
        [
            # 1. Establish simple existential reach tasks.
            RandomCurriculumStage(sampler=reach, threshold=0.9),
            # 2. Emphasize sequential reach while retaining simple reach.
            MultiRandomStage(
                stages=[random_stage(sequential_reach), random_stage(reach)],
                probs=[0.80, 0.20],
                threshold=0.95,
            ),
            # 3. Add reach-avoid with balanced replay of both earlier families.
            MultiRandomStage(
                stages=[
                    random_stage(reach_avoid),
                    random_stage(sequential_reach),
                    random_stage(reach),
                ],
                probs=[0.70, 0.15, 0.15],
                threshold=0.95,
            ),
            # 4. Add sequential reach-avoid and replay every prior family.
            MultiRandomStage(
                stages=[
                    random_stage(sequential_reach_avoid),
                    random_stage(reach_avoid),
                    random_stage(sequential_reach),
                    random_stage(reach),
                ],
                probs=[0.70, 0.10, 0.10, 0.10],
                threshold=0.9,
            ),
            # 5. Add weak-next sequences with 30% replay.
            MultiRandomStage(
                stages=[
                    random_stage(weak_next),
                    random_stage(sequential_reach_avoid),
                    random_stage(reach_avoid),
                    random_stage(sequential_reach),
                    random_stage(reach),
                ],
                probs=[0.70, 0.075, 0.075, 0.075, 0.075],
                threshold=0.9,
            ),
            # 6. Add a safety-guarantee conjunction with 40% replay.
            MultiRandomStage(
                stages=[
                    random_stage(safety_guarantee),
                    random_stage(weak_next),
                    random_stage(sequential_reach_avoid),
                    random_stage(reach_avoid),
                    random_stage(sequential_reach),
                    random_stage(reach),
                ],
                probs=[0.60, 0.08, 0.08, 0.08, 0.08, 0.08],
                threshold=0.9,
            ),
            # 7. Train indefinitely on a broad mixture of all seven families.
            MultiRandomStage(
                stages=[
                    random_stage(multi_safety_guarantee),
                    random_stage(safety_guarantee),
                    random_stage(weak_next),
                    random_stage(sequential_reach_avoid),
                    random_stage(reach_avoid),
                    random_stage(sequential_reach),
                    random_stage(reach),
                ],
                probs=[0.30, 0.15, 0.15, 0.15, 0.10, 0.10, 0.05],
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
    """Create fixed obligation examples for fast end-to-end validation.

    The final two stages exercise conjunctions of existential guarantees and
    universal safety obligations, ordered after the simpler task families.
    """

    propositions = list(env.propositions)
    required_propositions = 4
    if len(propositions) < required_propositions:
        raise ValueError(
            "The validation curriculum requires at least four propositions."
        )
    p, q, r, s = propositions[:required_propositions]
    formulas = [
        f"E(F({p}))",
        f"E(F({p} & F({q})))",
        f"E((!{q}) U {p})",
        f"E((!{q}) U ({p} & ((!{r}) U {s})))",
        f"E(F({p} & N({q})))",
        f"E(F({p})) & A(G(!{q}))",
        f"E(F({p}) & F({q})) & A(G(!({r} | {s})))",
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
