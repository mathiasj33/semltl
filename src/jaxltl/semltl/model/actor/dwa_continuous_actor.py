"""Continuous actor for deterministic automata without epsilon actions."""

from collections.abc import Callable

import distrax
import jax
from jax.nn.initializers import Initializer

from jaxltl.rl.actor.continuous_actor import ContinuousActor


class DWAContinuousActor(ContinuousActor):
    """SemLTL-compatible actor that emits only environment actions.

    ``SemLTLModel`` supplies epsilon-state features and a mask to every actor.
    A DWA has no epsilon transitions, so this actor accepts those arguments for
    interface compatibility but deliberately ignores them.
    """

    def __init__(
        self,
        in_size: int,
        action_dim: int,
        hidden_sizes: list[int],
        state_dependent_std: bool = True,
        hidden_activation: Callable[[jax.Array], jax.Array] = jax.nn.relu,
        output_activation: Callable[[jax.Array], jax.Array] = jax.nn.tanh,
        weight_init: Initializer | None = jax.nn.initializers.orthogonal(),  # noqa
        bias_init: Initializer | None = jax.nn.initializers.zeros,
        *,
        key: jax.Array,
    ):
        super().__init__(
            in_size=in_size,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            use_epsilon=False,
            state_dependent_std=state_dependent_std,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            weight_init=weight_init,
            bias_init=bias_init,
            key=key,
        )

    def __call__(
        self,
        features: jax.Array,
        _epsilon_features: jax.Array,
        epsilon_mask: jax.Array,
    ) -> distrax.Distribution:
        return super().__call__(features, epsilon_mask)
