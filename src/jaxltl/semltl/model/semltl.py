from math import prod
from typing import NamedTuple, override

import distrax
import hydra
import jax
import jax.numpy as jnp
from equinox import nn
from jaxtyping import PyTree
from omegaconf import DictConfig

from jaxltl.environments import spaces
from jaxltl.environments.spaces import Space
from jaxltl.networks.conv_net import ConvNet
from jaxltl.networks.mlp import MLP
from jaxltl.rl.actor.actor import Actor
from jaxltl.rl.actor_critic import ActorCritic


class SemLTLModel(ActorCritic):
    env_net: MLP | ConvNet
    actor: Actor
    critic: MLP
    projection: nn.Linear

    _flatten_features: bool

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        act_space: Space,
        key: jax.Array,
        **kwargs,
    ):
        config = DictConfig(kwargs)
        key, env_key = jax.random.split(key)
        is_conv = "ConvNet" in config.env_net._target_
        params = {"obs_shape": obs_shape} if is_conv else {"in_size": prod(obs_shape)}
        self.env_net = hydra.utils.instantiate(config.env_net, **params, key=env_key)
        self._flatten_features = not is_conv
        actor_key, critic_key = jax.random.split(key)
        joint_dim = self.env_net.output_size + config.embedding_dim
        params = (
            {"num_actions": act_space.n}
            if isinstance(act_space, spaces.Discrete)
            else {"action_dim": act_space.shape[0]}
        )
        self.actor = hydra.utils.instantiate(
            config.actor, in_size=joint_dim, **params, key=actor_key
        )
        self.critic = hydra.utils.instantiate(
            config.critic,
            in_size=joint_dim,
            out_size=1,
            final_layer_activation=False,
            key=critic_key,
        )
        self.projection = nn.Linear(
            in_features=config.semantic.embedding_size,
            out_features=config.embedding_dim,
            key=key,
        )

    @override
    def _get_action(
        self, features: tuple[jax.Array, jax.Array], obs: PyTree
    ) -> distrax.Distribution:
        return self.actor(*features, obs.epsilon_mask)  # type: ignore

    @override
    def _get_value(self, features: tuple[jax.Array, jax.Array]) -> jax.Array:
        value = jax.vmap(self.critic)(features[0])
        return value.squeeze(-1)

    @override
    def _compute_common_features(self, obs: PyTree) -> tuple[jax.Array, jax.Array]:
        """Computes observation features and state embeddings for current and epsilon
        states.

        Returns:
            A tuple of:
                - state embedding: (batch_size, embedding_dim)
                - epsilon state embeddings: (batch_size, max_eps_transitions, embedding_dim)
        """
        x = (
            self.flatten_features(obs.features)
            if self._flatten_features
            else obs.features.features
            if hasattr(obs.features, "features")
            else obs.features
        )
        x = jax.vmap(self.env_net)(x)
        emb = jax.vmap(self.projection)(obs.embedding)
        # obs.epsilon_embeddings: (batch_size,max_eps_transitions,embedding_dim)
        epsilon_emb = jax.vmap(jax.vmap(self.projection))(obs.epsilon_embeddings)
        x_tiled = jnp.tile(x[:, None, :], (1, epsilon_emb.shape[1], 1))
        return jnp.concatenate([x, emb], axis=-1), jnp.concatenate(
            [x_tiled, epsilon_emb], axis=-1
        )

    @staticmethod
    def flatten_features(features: NamedTuple) -> jax.Array:
        return jnp.concatenate(
            [v.reshape(v.shape[0], -1) for v in jax.tree.leaves(features)], axis=-1
        )
