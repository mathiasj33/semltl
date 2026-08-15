"""Utility functions for batching formulas into SemanticJaxLDBAs."""

import numpy as np
from tqdm import tqdm

from jaxltl.environments.environment import Environment
from jaxltl.environments.wrappers.wrapper import EnvWrapper
from jaxltl.ltl.automata.ltl2ldba import ltlfplus2dba_fishsemml
from jaxltl.ltl.logic.assignment import Assignment
from jaxltl.semltl.utils.jax_semantic_ldba import JaxSemanticLDBA
from jaxltl.utils import memory

# from jaxltl.ltl.automata.ltl2ldba import ltl2ldba_semml


def preprocess_formulas(
    formulas: list[str], env: Environment | EnvWrapper
) -> JaxSemanticLDBA:
    """Converts a list of LTL formulas into a batched SemanticJaxLDBA.

    Args:
        formulas: A list of formulas.

    Returns:
        A batched SemanticJaxLDBA.
    """

    ldbas = [
        build_semantic_ldba(formula, env.propositions, tuple(env.assignments))
        for formula in tqdm(formulas, desc="Building Semantic LDBAs")
    ]
    return JaxSemanticLDBA.from_ldbas(ldbas, env)


@memory.cache
def build_semantic_ldba(
    formula: str,
    propositions: tuple[str, ...],
    assignments: tuple[Assignment, ...],
):
    # ldba = ltl2ldba_semml(formula, propositions, assignments, use_attention=True)
    dba = ltlfplus2dba_fishsemml(formula, propositions, assignments)
    for state in dba.states:
        info = dba.state_to_info[state]
        info["embedding"] = get_semantic_embedding(info)
    dba.prune(list(assignments))
    dba.complete_sink_state()
    dba.compute_sccs()
    return dba


def get_semantic_embedding(state_info: dict) -> np.ndarray:
    """Extracts the semantic embedding from the state info dictionary.

    Args:
        state_info: A dictionary containing information about the state.

    Returns:
        A numpy array representing the semantic embedding of the state.
    """
    embedding = np.asarray(state_info["formula_embedding"], dtype=np.float32)
    return embedding

    # expected_size = 167
    # if embedding.shape != (expected_size,):
    #         raise ValueError(
    #             f"Expected embedding size {expected_size}, got {embedding.shape}"
    #         )

    # return embedding


# def get_semantic_embedding(state_info: dict) -> np.ndarray:
#     embeddings = state_info["embeddings"]
#     if state_info["component"] == "initial":
#         embedding = embeddings["formula_embedding"]
#         embedding += [0.0] * len(embedding)  # empty breakpoint embedding
#     else:
#         embedding = embeddings["master_formula_embedding"]
#         embedding += embeddings["breakpoint_formula_embedding"]
#     return np.array(embedding, dtype=np.float32)
