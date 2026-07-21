from collections.abc import Callable
from typing import override

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import nn

from jaxltl.networks.callable_module import CallableModule
from jaxltl.networks.mlp import MLP


class RGBLidarConvNet(CallableModule):
    """
    A network for RGBZoneEnv with rgb_lidar exteroception.

    It consists of a 1D CNN for lidar data, an MLP for proprioceptive data,
    and a fusion MLP to combine their outputs.
    """

    lidar_cnn: nn.Sequential
    proprio_mlp: MLP
    fusion_mlp: MLP
    output_size: int

    def __init__(
        self,
        channels: list[int],
        kernel_size: int,
        proprio_hidden_sizes: list[int],
        proprio_out_size: int,
        fusion_hidden_sizes: list[int],
        fusion_out_size: int,
        activation: Callable[[jax.Array], jax.Array] = jax.nn.relu,
        *,
        key: jax.Array,
        **kwargs,
    ):
        """
        Initializes the RGBLidarConvNet.

        Args:
            obs_shape: A RGBLidarObsSpace.Shape containing the shapes of the observations.
            channels: List of output channels for each Conv1D layer.
            kernel_size: Kernel size for the Conv1D layers.
            proprio_hidden_sizes: Hidden layer sizes for the proprioception MLP.
            proprio_out_size: The output size of the proprioception MLP.
            fusion_hidden_sizes: Hidden layer sizes for the fusion MLP.
            fusion_out_size: The final output size of the network.
            activation: The activation function to use.
            key: JAX random key.
        """
        lidar_key, proprio_key, fusion_key = jax.random.split(key, 3)

        # 1. Lidar CNN
        num_layers = 1
        num_features = 5
        num_bins, c = 32, num_layers * num_features
        channels = [c, *channels]
        cnn_layers = []
        current_seq_len = num_bins
        for i in range(len(channels) - 1):
            lidar_key, subkey = jax.random.split(lidar_key)
            cnn_layers.append(
                nn.Conv1d(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2,
                    padding_mode="CIRCULAR",
                    key=subkey,
                )
            )
            cnn_layers.append(nn.Lambda(activation))
            cnn_layers.append(nn.AvgPool1d(kernel_size=2, stride=2))
            current_seq_len //= 2

        self.lidar_cnn = nn.Sequential(cnn_layers)
        cnn_output_size = current_seq_len * channels[-1]

        # 2. Proprioception MLP
        proprio_in_size = 5
        self.proprio_mlp = MLP(
            in_size=proprio_in_size,
            hidden_sizes=proprio_hidden_sizes,
            out_size=proprio_out_size,
            activation=activation,
            key=proprio_key,
        )

        # 3. Fusion MLP
        fusion_in_size = cnn_output_size + proprio_out_size
        self.fusion_mlp = MLP(
            in_size=fusion_in_size,
            hidden_sizes=fusion_hidden_sizes,
            out_size=fusion_out_size,
            activation=activation,
            key=fusion_key,
        )
        self.output_size = fusion_out_size

    @override
    @eqx.filter_jit
    def __call__(self, x):
        """Forward pass through the network."""
        # Lidar features
        lidar_feat = x.rgb_lidar.astype(jnp.float32)
        lidar_feat = lidar_feat.reshape(
            lidar_feat.shape[0], -1
        )  # (num_bins, num_layers*5)
        lidar_feat = jnp.transpose(lidar_feat, (1, 0))  # WC to CW
        lidar_out = self.lidar_cnn(lidar_feat)
        lidar_out = lidar_out.flatten()

        # Proprioceptive features
        proprio_feat = jnp.concatenate(
            [
                x.acceleration,
                x.velocity,
                x.angular_velocity,
            ],
            axis=-1,
        ).astype(jnp.float32)
        proprio_out = self.proprio_mlp(proprio_feat)

        # Fusion
        fused_input = jnp.concatenate([lidar_out, proprio_out])
        final_output = self.fusion_mlp(fused_input)

        return final_output
