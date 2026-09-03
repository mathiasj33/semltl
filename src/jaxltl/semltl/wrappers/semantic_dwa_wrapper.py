"""Semantic DWA wrapper for FishSemML Büchi automata."""

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
    """Run a FishSemML DWA using its Büchi transition acceptance.

    The observation shape remains compatible with the existing SemLTL model,
    but the DWA-specific actor emits only an environment action. All epsilon
    masks are false because a DWA has no epsilon transitions.

    An accepting transition gives reward ``+1``, a transition into a rejecting
    bottom component gives ``-1``, and every other transition gives ``0``.
    Accepting and rejecting bottom components terminate early because the
    Büchi result is irrevocably decided once either is entered. Other accepting
    transitions remain non-terminal and can produce recurring rewards.
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
                "dwa_accepting_transition": jnp.asarray(False),
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
        action: jax.Array,
        params: TEnvParams,
    ) -> EnvTransition[SemanticLDBAWrapperState, TObsFeatures]:
        transition = super().step(key, state, action, params)

        assignment = self._env.map_assignment_to_index(transition.propositions)
        next_dwa_state, is_accepting_transition = state.ldba.get_next_state(
            state.ldba_state, assignment
        )
        is_accepting_state = state.ldba.accepting_states[next_dwa_state]
        is_accepting_sink = state.ldba.accepting_sink_states[next_dwa_state]
        is_rejecting_sink = state.ldba.rejecting_sink_states[next_dwa_state]

        # FishSemML marks transitions according to their source state. Entering
        # an accepting bottom component must nevertheless yield terminal
        # success: all of its future transitions would be accepting. Conversely,
        # entering a rejecting bottom component is terminal failure even if the
        # source transition was accepting once; one accepting visit cannot
        # satisfy a Büchi condition followed by permanent rejection.
        reward = jnp.where(
            is_rejecting_sink,
            -1.0,
            jnp.where(
                is_accepting_transition | is_accepting_sink,
                1.0,
                0.0,
            ),
        )

        info = {
            **transition.info,
            # This means that acceptance has become irrevocable, not that a
            # finite episode ending in any accepting state has been accepted.
            "satisfied": is_accepting_sink,
            "dwa_accepting_transition": is_accepting_transition,
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
