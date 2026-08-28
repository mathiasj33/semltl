"""Finite-trace semantic DWA wrapper for FishSemML automata."""

from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxltl.environments.environment import Environment, EnvObservation, EnvTransition
from jaxltl.environments.wrappers import EnvWrapper
from jaxltl.ltl2action.wrappers.curriculum_wrapper import CurriculumResetOptions
from jaxltl.semltl.utils.jax_semantic_ldba import JaxSemanticLDBA
from jaxltl.semltl.wrappers.semantic_ldba_wrapper import (
    SemanticLDBAObservation,
    SemanticLDBAWrapperState,
)


class SemanticDWAWrapper[
    TEnvParams,
    TObsFeatures: NamedTuple,
](EnvWrapper[TEnvParams, TObsFeatures, CurriculumResetOptions]):
    """Use FishSemML state acceptance with finite-trace semantics.

    The observation and action shapes intentionally remain compatible with the
    existing SemLTL model. The epsilon part of an action is ignored and all
    epsilon masks are false because a DWA has no epsilon transitions.

    Logical satisfaction is decided when the underlying finite trace ends. A
    rejecting sink fails immediately. The exact result is exposed as
    ``info["satisfied"]`` and must be used for evaluation instead of
    interpreting a positive accumulated reward as success.
    """

    def __init__(
        self,
        env: (
            EnvWrapper[TEnvParams, TObsFeatures, CurriculumResetOptions]
            | Environment[Any, TEnvParams, TObsFeatures, CurriculumResetOptions]
        ),
    ):
        super().__init__(env)

    def _observation(
        self,
        obs: EnvObservation[TObsFeatures],
        dwa: JaxSemanticLDBA,
        dwa_state: jax.Array,
    ) -> SemanticLDBAObservation[TObsFeatures]:
        embedding = dwa.get_embedding(dwa_state)
        # JaxSemanticLDBA keeps one padded epsilon slot. It remains disabled,
        # preserving checkpoint-compatible observation shapes.
        epsilon_embeddings = jnp.zeros_like(dwa.get_epsilon_embeddings(dwa_state))
        epsilon_mask = jnp.zeros(
            dwa.epsilon_transitions.shape[1], dtype=jnp.bool
        )
        return SemanticLDBAObservation.from_obs(
            obs, embedding, epsilon_embeddings, epsilon_mask
        )

    @eqx.filter_jit
    def reset(
        self,
        key: jax.Array,
        state: SemanticLDBAWrapperState | None,
        params: TEnvParams,
        options: CurriculumResetOptions | None = None,
    ) -> tuple[SemanticLDBAWrapperState, SemanticLDBAObservation[TObsFeatures]]:
        assert options is not None, (
            "CurriculumResetOptions must be provided to reset."
        )
        env_state, obs = super().reset(key, state, params, options)
        propositions = self._env.compute_propositions(env_state, params)
        state = SemanticLDBAWrapperState(
            state=env_state,
            ldba=options.task,
            ldba_state=options.task.initial_state,
            obs=obs,
            propositions=propositions,
            # Keep the reset and step states structurally identical. JAX control
            # flow (e.g. AutoResetWrapper's lax.cond) requires both branches to
            # have the same pytree keys, even before these values are meaningful.
            info={
                "satisfied": jnp.asarray(False),
                "dwa_accepting": jnp.asarray(False),
                "dwa_accepting_sink": jnp.asarray(False),
                "dwa_rejecting_sink": jnp.asarray(False),
            },
        )
        return state, self._observation(obs, state.ldba, state.ldba_state)

    @eqx.filter_jit
    def cheap_reset(self, *args, **kwargs):
        raise NotImplementedError()

    @eqx.filter_jit
    def step(
        self,
        key: jax.Array,
        state: SemanticLDBAWrapperState,
        action: tuple[jax.Array, jax.Array],
        params: TEnvParams,
    ) -> EnvTransition[SemanticLDBAWrapperState, TObsFeatures]:
        env_action, _epsilon_action = action
        transition = super().step(key, state, env_action, params)

        assignment = self._env.map_assignment_to_index(transition.propositions)
        next_dwa_state = state.ldba.transitions[state.ldba_state, assignment]
        is_accepting_state = state.ldba.accepting_states[next_dwa_state]
        is_accepting_sink = state.ldba.accepting_sink_states[next_dwa_state]
        is_rejecting_sink = state.ldba.rejecting_sink_states[next_dwa_state]

        trace_ended = transition.done
        satisfied = is_accepting_sink | (
            trace_ended & is_accepting_state & ~is_rejecting_sink
        )
        failed = is_rejecting_sink | (trace_ended & ~is_accepting_state)
        reward = jnp.where(failed, -1.0, jnp.where(satisfied, 1.0, 0.0))

        info = {
            **transition.info,
            "satisfied": satisfied,
            "dwa_accepting": is_accepting_state,
            "dwa_accepting_sink": is_accepting_sink,
            "dwa_rejecting_sink": is_rejecting_sink,
        }
        new_state = SemanticLDBAWrapperState(
            state=transition.state,
            ldba=state.ldba,
            ldba_state=next_dwa_state,
            obs=transition.observation,
            propositions=transition.propositions,
            info=info,
        )
        observation = self._observation(
            transition.observation, state.ldba, next_dwa_state
        )
        terminal_observation = self._observation(
            transition.terminal_observation, state.ldba, next_dwa_state
        )
        return EnvTransition(
            state=new_state,
            observation=observation,
            reward=reward,
            terminated=(
                transition.terminated | is_accepting_sink | is_rejecting_sink
            ),
            truncated=transition.truncated,
            terminal_observation=terminal_observation,
            propositions=transition.propositions,
            info=info,
        )
