import random
from pathlib import Path

from jaxltl.environments.environment import Environment
from jaxltl.environments.wrappers.wrapper import EnvWrapper
from jaxltl.ltl2action.curriculum.curriculum import (
    Curriculum,
    RandomCurriculumStage,
    Sampler,
)
from jaxltl.semltl.curriculum.batching import SemanticLDBABatcher


def make(env: Environment | EnvWrapper, load_path: Path | None = None) -> Curriculum:
    return Curriculum(
        [
            # 1. Simple reach tasks
            RandomCurriculumStage(
                sampler=ConveyorFormulaSampler(),
                threshold=None,
            ),
        ],
        num_samples=1000,
        batcher=SemanticLDBABatcher(),
        env=env,
        load_path=load_path,
    )


class ConveyorFormulaSampler(Sampler[str]):
    """Samples formulas specific to the conveyor world."""

    def sample(self) -> str:
        if random.random() < 0.5:
            return "F(parcel & (F wrench))"
        else:
            return "F(parcel & (F hammer))"
