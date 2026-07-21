from pathlib import Path

from jaxltl.environments.environment import Environment
from jaxltl.environments.wrappers.wrapper import EnvWrapper
from jaxltl.ltl2action.curriculum.curriculum import (
    Curriculum,
    MultiRandomStage,
    RandomCurriculumStage,
)
from jaxltl.ltl2action.curriculum.simple_samplers import (
    SimpleGFSampler,
    SimpleReachAvoidFormulaSampler,
)
from jaxltl.semltl.curriculum.batching import SemanticLDBABatcher


def make(env: Environment | EnvWrapper, load_path: Path | None = None) -> Curriculum:
    return Curriculum(
        [
            # 1. Simple reach tasks
            RandomCurriculumStage(
                sampler=SimpleReachAvoidFormulaSampler(
                    depth=1,
                    reach=1,
                    avoid=(0, 1),
                    propositions=list(env.propositions),
                ),
                threshold=0.9,
            ),
            # 2. Reach tasks of depth 2
            RandomCurriculumStage(
                sampler=SimpleReachAvoidFormulaSampler(
                    depth=1,
                    reach=(1, 2),
                    avoid=(0, 2),
                    propositions=list(env.propositions),
                ),
                threshold=0.95,
            ),
            # 3. Simple reach-avoid tasks
            RandomCurriculumStage(
                sampler=SimpleReachAvoidFormulaSampler(
                    depth=2,
                    reach=(1, 2),
                    avoid=(1, 2),
                    propositions=list(env.propositions),
                ),
                threshold=0.95,
            ),
            # 4. Reach-avoid tasks of depth 2, GF tasks
            MultiRandomStage(
                [
                    RandomCurriculumStage(
                        sampler=SimpleReachAvoidFormulaSampler(
                            depth=3,
                            reach=(1, 2),
                            avoid=(0, 3),
                            propositions=list(env.propositions),
                        ),
                        threshold=None,
                    ),
                    RandomCurriculumStage(
                        sampler=SimpleGFSampler(
                            reach=(2, 4),
                            avoid=(0, 2),
                            propositions=list(env.propositions),
                        ),
                        threshold=None,
                    ),
                ],
                probs=[0.5, 0.5],
                threshold=None,
            ),
        ],
        num_samples=10_000,
        batcher=SemanticLDBABatcher(),
        env=env,
        load_path=load_path,
    )
