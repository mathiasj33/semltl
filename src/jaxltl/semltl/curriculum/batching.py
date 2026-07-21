from typing import override

from jaxltl.environments.environment import Environment
from jaxltl.environments.wrappers.wrapper import EnvWrapper
from jaxltl.ltl2action.curriculum.curriculum import SampleBatcher
from jaxltl.semltl.utils.jax_semantic_ldba import JaxSemanticLDBA
from jaxltl.semltl.utils.preprocessing import preprocess_formulas


class SemanticLDBABatcher(SampleBatcher[str, JaxSemanticLDBA]):
    """Batches formulas into a JaxSemanticLDBA."""

    @override
    @staticmethod
    def batch(
        samples: list[str],
        env: Environment | EnvWrapper,
    ) -> JaxSemanticLDBA:
        return preprocess_formulas(samples, env)
