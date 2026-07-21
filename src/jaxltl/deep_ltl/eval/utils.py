"""Utility functions for evaluation scripts."""

import jax.numpy as jnp

from jaxltl.deep_ltl.reach_avoid import path_search
from jaxltl.deep_ltl.reach_avoid.reach_avoid_sequence import ReachAvoidSequence
from jaxltl.environments.environment import Environment
from jaxltl.environments.wrappers.wrapper import EnvWrapper
from jaxltl.ltl.automata import ltl2ldba, ltl2ldba_semml
from jaxltl.ltl.automata.jax_ldba import JaxLDBA
from jaxltl.ltl.automata.ldba import LDBA
from jaxltl.ltl.logic.assignment import Assignment
from jaxltl.utils import memory


@memory.cache
def compute_cached_ldba_and_sequences(
    formula: str, propositions: tuple[str, ...], assignments: tuple[Assignment, ...]
) -> tuple[LDBA, dict[int, list[ReachAvoidSequence]]]:
    ldba = ltl2ldba_semml(formula, propositions, assignments)
    print("Pruning LDBA...")
    ldba.prune(list(assignments))
    print("Completing sink state...")
    ldba.complete_sink_state()
    print("Computing SCCs...")
    ldba.compute_sccs()
    return ldba, path_search.compute_sequences(ldba, num_loops=2, verbose=True)


def _build_ldba(formula: str, env: Environment | EnvWrapper):
    ldba = ltl2ldba(formula, env.propositions)
    ldba.prune(env.assignments)
    ldba.complete_sink_state()
    ldba.compute_sccs()
    return ldba


def _batch_ldbas(ldbas: list[JaxLDBA]) -> JaxLDBA:
    """Batch multiple JaxLDBAs into a single JaxLDBA with an added batch dimension."""

    num_states = jnp.array([ldba.num_states for ldba in ldbas], dtype=jnp.int32)
    max_num_states = jnp.max(num_states)
    batch_size = len(ldbas)
    num_assignments = ldbas[0].transitions.shape[1] - 1

    transitions = -jnp.ones(
        (batch_size, max_num_states, num_assignments + 1), dtype=jnp.int32
    )
    accepting = jnp.zeros((batch_size, max_num_states, num_assignments), dtype=bool)
    sink_states = jnp.zeros((batch_size, max_num_states), dtype=bool)
    initial_states = jnp.zeros((batch_size,), dtype=jnp.int32)

    for i, ldba in enumerate(ldbas):
        transitions = transitions.at[i, : ldba.num_states, :].set(ldba.transitions)
        accepting = accepting.at[i, : ldba.num_states, :].set(ldba.accepting)
        sink_states = sink_states.at[i, : ldba.num_states].set(ldba.sink_states)
        initial_states = initial_states.at[i].set(ldba.initial_state)

    return JaxLDBA(
        num_states=num_states,
        initial_state=initial_states,
        transitions=transitions,
        accepting=accepting,
        sink_states=sink_states,
        finite=jnp.array([ldba.finite for ldba in ldbas]),
    )
