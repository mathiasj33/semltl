from typing import override

import jax
import jax.numpy as jnp

from jaxltl.deep_ltl.wrappers.ldba_wrapper import LDBAWrapperState
from jaxltl.environments.environment import EnvObservation
from jaxltl.environments.wrappers.wrapper import EnvWrapper
from jaxltl.eval.agent import Agent
from jaxltl.rl.actor_critic import ActorCritic
from jaxltl.semltl.wrappers.semantic_ldba_wrapper import SemanticLDBAWrapperState


class SemLTLAgent(Agent[LDBAWrapperState]):
    """Agent for SemLTL that stores info about the number of explored LDBA states."""

    visited: jax.Array  # (num_envs,num_states) or None

    @override
    @classmethod
    def instantiate(
        cls,
        model: ActorCritic,
    ) -> "Agent":
        return cls(model, None)  # type: ignore

    @override
    def update(
        self,
        obsv: EnvObservation,
        state: SemanticLDBAWrapperState,
        props: jax.Array,
        env: EnvWrapper,
    ) -> "SemLTLAgent":
        if self.visited is None:
            visited = jnp.zeros(
                (
                    state.ldba_state.shape[0],
                    state.ldba.transitions.shape[1],
                ),
                dtype=bool,
            )
        else:
            visited = self.visited
        new_visited = visited.at[
            jnp.arange(state.ldba_state.shape[0]), state.ldba_state
        ].set(True)
        return SemLTLAgent(model=self.model, visited=new_visited)

    @override
    def info(self) -> dict:
        return {"num_visited_ldba_states": jnp.sum(self.visited, axis=-1)}
