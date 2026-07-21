"""An adaptation of the zone environment introduced by LTL2Action (Vaezipoor et al., 2021).
Instead of 4 colours with 2 zones each, this environment supports 8 zones with distinct colours.
It also supports various different lidar types.

The environment simulates a point-mass agent moving in a 2D plane. The agent
applies a forward force aligned with its current heading and can control its
angular velocity. The world contains colored zones that the agent can enter.
The agent is equipped with a lidar sensor that detects the distance to the
nearest zone of each color in a set of evenly spaced angular bins.
"""

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, override

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax

from jaxltl.environments import environment, spaces
from jaxltl.ltl.logic.assignment import Assignment

from .plotter import draw_trajectories

if TYPE_CHECKING:
    from jaxltl.environments.renderer.renderer import BaseRenderer

_EPS = 1e-8


@dataclass(frozen=True)
class EnvParams(environment.EnvParams):
    # World
    world_size: float
    spawn_size: float
    # Zones
    zone_radius: float
    zones_per_color: int
    keepout_radius: float
    # Lidar
    num_lidar_bins: int
    exp_gain: float
    lidar_type: Literal["lidar", "rgb_lidar", "2_rgb_lidar", "privileged"]
    # Physics
    dt: float
    drag: float
    max_speed: float
    max_force: float
    max_angular_velocity: float


class EnvState(eqx.Module):
    # Physics
    position: jax.Array  # shape: (2,)
    velocity: jax.Array  # shape: (2,)
    angle: jax.Array  # shape: ()
    angular_velocity: jax.Array  # float
    acceleration: jax.Array  # shape: (2,)
    # Zones (static for an episode)
    zone_centers: jax.Array  # shape: (N, 2)
    zone_colors: jax.Array  # shape: (N,) int in [0, C)


class RGBLidarObsFeatures(NamedTuple):
    acceleration: jax.Array  # shape: (2,)
    velocity: jax.Array  # shape: (2,)
    angular_velocity: jax.Array  # shape: (1,)
    rgb_lidar: jax.Array  # shape: (num_bins, 5) -> (r,g,b,intensity,detected)


class LidarObsFeatures(NamedTuple):
    acceleration: jax.Array  # shape: (2,)
    velocity: jax.Array  # shape: (2,)
    angular_velocity: jax.Array  # shape: (1,)
    lidar: jax.Array  # shape: (num_colors, num_bins)


class PrivilegedObsFeatures(NamedTuple):
    acceleration: jax.Array  # shape: (2,)
    velocity: jax.Array  # shape: (2,)
    angular_velocity: jax.Array  # shape: (1,)
    zone_vecs: jax.Array  # shape: (N, 3) -> (dist, angle_sin, angle_cos)


type ObsFeatures = RGBLidarObsFeatures | LidarObsFeatures | PrivilegedObsFeatures


class ResetOptions(NamedTuple):
    pass


class ZoneEnv8(environment.Environment[EnvState, EnvParams, ObsFeatures, ResetOptions]):
    default_params = EnvParams(
        max_steps_in_episode=1000,
        world_size=6.6,
        spawn_size=5.0,
        zone_radius=0.4,
        zones_per_color=1,
        keepout_radius=0.55,
        num_lidar_bins=32,
        lidar_type="rgb_lidar",
        exp_gain=0.5,
        dt=0.05,
        drag=0.08,
        max_speed=3.0,
        max_force=2.0,
        max_angular_velocity=3.0,
    )
    propositions = ("blue", "orange", "green", "red", "purple", "brown", "pink", "gray")
    rgbs = (
        jnp.array(
            [
                (31, 119, 180),
                (255, 127, 14),
                (44, 160, 44),
                (214, 39, 40),
                (148, 103, 189),
                (140, 86, 75),
                (227, 119, 194),
                (127, 127, 127),
            ],
            dtype=jnp.float32,
        )
        / 255.0
    )
    max_nodes: int = 1
    max_edges: int = 1

    def __init__(self, **kwargs):
        params = dataclasses.asdict(self.default_params) | kwargs
        super().__init__(
            default_params=EnvParams(**params),
            propositions=self.propositions,
            max_nodes=self.max_nodes,
            max_edges=self.max_edges,
        )
        assert len(self.propositions) == self.rgbs.shape[0], (
            "RGBs must match propositions"
        )

    @override
    def _observation_space(self, params: EnvParams) -> spaces.Space:
        if params.lidar_type == "lidar":
            shape = (5 + params.num_lidar_bins * len(self.propositions),)
        elif params.lidar_type == "rgb_lidar":
            shape = (5 + params.num_lidar_bins * 5,)
        elif params.lidar_type == "2_rgb_lidar":
            shape = (5 + params.num_lidar_bins * 2 * 5,)
        elif params.lidar_type == "privileged":
            shape = (5 + params.zones_per_color * len(self.propositions) * 3,)
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=shape,
            dtype=jnp.float32,
        )

    @override
    def _action_space(self, params: EnvParams) -> spaces.Space:
        return spaces.Box(
            low=jnp.array(
                [-params.max_force, -params.max_angular_velocity], dtype=jnp.float32
            ),
            high=jnp.array(
                [params.max_force, params.max_angular_velocity], dtype=jnp.float32
            ),
            shape=(2,),
            dtype=jnp.float32,
        )

    @override
    def _reset(
        self,
        key_angle: jax.Array,
        state: EnvState | None,
        params: EnvParams,
        options: ResetOptions | None = None,
    ) -> EnvState:
        key_zones, key_pos, key_angle = jax.random.split(key_angle, 3)
        centers, colors = self._sample_zones(key_zones, params)
        agent_pos = self._sample_agent_position(key_pos, params, centers)

        velocity = jnp.zeros(2, dtype=jnp.float32)
        acceleration = jnp.zeros(2, dtype=jnp.float32)
        angle = jax.random.uniform(key_angle, shape=(), minval=-jnp.pi, maxval=jnp.pi)
        angular_velocity = jnp.zeros((), dtype=jnp.float32)

        return EnvState(
            position=agent_pos,
            velocity=velocity,
            angle=angle,
            angular_velocity=angular_velocity,
            acceleration=acceleration,
            zone_centers=centers,
            zone_colors=colors,
        )

    @override
    def _cheap_reset(
        self,
        key: jax.Array,
        state: EnvState,
        params: EnvParams,
        options: ResetOptions | None = None,
    ) -> EnvState:
        raise NotImplementedError("Cheap reset is not implemented for ZoneEnv8.")

    def _sample_zones(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, jax.Array]:
        """Sample non-overlapping zone centers and assign colors.

        Returns (centers:(Z,2), colors:(Z,))
        """
        num_colors = len(self.propositions)
        total_zones = num_colors * params.zones_per_color
        minval = -params.spawn_size / 2 + params.keepout_radius
        maxval = params.spawn_size / 2 - params.keepout_radius
        keepout = params.keepout_radius

        centers0 = jnp.zeros((total_zones, 2), dtype=jnp.float32)
        colors = jnp.repeat(
            jnp.arange(num_colors, dtype=jnp.int32), params.zones_per_color
        )

        def cond_fun(carry):
            key, centers, count = carry
            return count < total_zones

        def body_fun(carry):
            key, centers, count = carry
            key, sub = jax.random.split(key)
            proposal = jax.random.uniform(sub, (2,), minval=minval, maxval=maxval)
            idxs = jnp.arange(total_zones)
            mask = idxs < count
            dists = jnp.linalg.norm(centers - proposal, axis=1)
            cond_ok = dists >= 2.0 * keepout
            all_ok = jnp.all(jnp.logical_or(~mask, cond_ok))
            centers = lax.cond(
                all_ok,
                lambda c: c.at[count].set(proposal),
                lambda c: c,
                centers,
            )
            count = count + jnp.where(all_ok, 1, 0)
            return key, centers, count

        key, centers, count = lax.while_loop(
            cond_fun, body_fun, (key, centers0, jnp.int32(0))
        )
        return centers, colors

    def _sample_agent_position(
        self, key: jax.Array, params: EnvParams, centers: jax.Array
    ) -> jax.Array:
        minval = -params.spawn_size / 2 + params.keepout_radius
        maxval = params.spawn_size / 2 - params.keepout_radius

        def agent_cond(carry):
            key, pos, it = carry
            dists = jnp.linalg.norm(centers - pos, axis=1)
            return jnp.logical_and(
                jnp.any(dists < params.keepout_radius * 2), it < 1000
            )

        def agent_body(carry):
            key, _pos, it = carry
            key, sub = jax.random.split(key)
            pos = jax.random.uniform(sub, (2,), minval=minval, maxval=maxval)
            return key, pos, it + 1

        key_init, key = jax.random.split(key)
        init_pos = jax.random.uniform(key_init, (2,), minval=minval, maxval=maxval)
        key, pos, _ = lax.while_loop(
            agent_cond, agent_body, (key, init_pos, jnp.int32(0))
        )
        return pos

    @override
    def _step(
        self,
        key: jax.Array,
        state: EnvState,
        action: jax.Array,
        params: EnvParams,
    ) -> tuple[EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        force = jnp.clip(
            action[0] * params.max_force, -params.max_force, params.max_force
        )
        target_angular_velocity = jnp.clip(
            action[1] * params.max_angular_velocity,
            -params.max_angular_velocity,
            params.max_angular_velocity,
        )
        heading = jnp.array([jnp.cos(state.angle), jnp.sin(state.angle)])
        acceleration = heading * force

        velocity = state.velocity + acceleration * params.dt
        velocity *= 1.0 - params.drag

        speed = jnp.linalg.norm(velocity)
        scaling_factor = jnp.clip(params.max_speed / speed, 0.0, 1.0)
        velocity: jax.Array = jnp.where(
            speed > params.max_speed, velocity * scaling_factor, velocity
        )

        position = state.position + velocity * params.dt

        angle = self._wrap_angle(state.angle + target_angular_velocity * params.dt)
        angular_velocity = target_angular_velocity

        reward = jnp.zeros((), dtype=jnp.float32)
        half_size = params.world_size / 2.0
        terminated = jnp.any(jnp.abs(position) > half_size)

        next_state = EnvState(
            position=position,
            velocity=velocity,
            angle=angle,
            angular_velocity=angular_velocity,
            acceleration=acceleration,
            zone_centers=state.zone_centers,
            zone_colors=state.zone_colors,
        )
        return next_state, reward, terminated, {}

    @staticmethod
    def _wrap_angle(angle: jax.Array) -> jax.Array:
        """Wrap angles to the (-pi, pi] interval."""
        return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

    def _get_true_max_speed(self, params: EnvParams) -> jax.Array:
        """Calculate the true maximum speed considering drag."""
        steady_state_speed = (params.max_force * params.dt * (1.0 - params.drag)) / (
            params.drag + _EPS
        )
        return jnp.minimum(steady_state_speed, params.max_speed)

    @override
    def _compute_obs(self, state: EnvState, params: EnvParams) -> ObsFeatures:
        """Compute the observation for a given state."""
        # Create rotation matrix to transform from world to agent frame
        c, s = jnp.cos(state.angle), jnp.sin(state.angle)
        rot = jnp.array([[c, s], [-s, c]])

        # Rotate and normalize acceleration and velocity
        acceleration = jnp.dot(rot, state.acceleration) / params.max_force
        velocity = jnp.dot(rot, state.velocity) / self._get_true_max_speed(params)
        angular_velocity = state.angular_velocity / params.max_angular_velocity

        if params.lidar_type == "lidar":
            lidar = self._compute_lidar(state, params)
            return LidarObsFeatures(
                acceleration=acceleration,
                velocity=velocity,
                angular_velocity=angular_velocity.reshape(1),
                lidar=lidar,
            )
        elif params.lidar_type == "rgb_lidar":
            lidar = self._compute_rgb_lidar(state, params)
            return RGBLidarObsFeatures(
                acceleration=acceleration,
                velocity=velocity,
                angular_velocity=angular_velocity.reshape(1),
                rgb_lidar=lidar,
            )
        elif params.lidar_type == "2_rgb_lidar":
            lidar = self._compute_2_rgb_lidar(state, params)
            return RGBLidarObsFeatures(
                acceleration=acceleration,
                velocity=velocity,
                angular_velocity=angular_velocity.reshape(1),
                rgb_lidar=lidar,
            )
        elif params.lidar_type == "privileged":
            zone_vecs = self._compute_zone_vecs(state, params)
            return PrivilegedObsFeatures(
                acceleration=acceleration,
                velocity=velocity,
                angular_velocity=angular_velocity.reshape(1),
                zone_vecs=zone_vecs,
            )
        else:
            raise ValueError(f"Unknown lidar type: {params.lidar_type}")

    def _compute_lidar(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Compute per-color lidar distances with evenly spaced bins around the agent.

        Returns an array of shape (C, num_bins) with distances in world units.
        """
        pos = state.position  # (2,)
        bin_size = 2.0 * jnp.pi / params.num_lidar_bins
        heading = jnp.array([jnp.cos(state.angle), jnp.sin(state.angle)])  # (2,)

        centers = state.zone_centers  # (N,2)
        colors = state.zone_colors  # (N,)

        def zone_sensor_binned(zone_pos: jax.Array) -> jax.Array:
            """Compute the sensor of a single zone.

            Returns: (num_bins,)"""

            direction = zone_pos - pos  # (2,)
            dist: float = jnp.linalg.norm(direction)  # ()
            sensor = jnp.exp(-params.exp_gain * dist)
            direction = direction / (dist + _EPS)  # (2,)
            dotp = jnp.dot(heading, direction)
            cross = jnp.cross(heading, direction)
            angle = jnp.arctan2(cross, dotp) % (2.0 * jnp.pi)
            bin_idx = jnp.floor(angle / bin_size).astype(jnp.int32)
            bin_angle = bin_size * bin_idx
            bins = jnp.zeros((params.num_lidar_bins,), dtype=jnp.float32)
            alias = (angle - bin_angle) / bin_size
            bins = bins.at[bin_idx].set(sensor)
            bins = bins.at[(bin_idx + 1) % params.num_lidar_bins].set(sensor * alias)
            bins = bins.at[(bin_idx - 1) % params.num_lidar_bins].set(
                sensor * (1.0 - alias)
            )
            return bins

        sensors = jax.vmap(zone_sensor_binned, in_axes=0)(centers)  # (N, num_bins)

        def compute_color_lidar(color_id: jax.Array) -> jax.Array:
            """Compute lidar for a single color."""
            mask_color = colors == color_id  # (num_zones,)
            sensors_color = jnp.where(
                mask_color[:, None], sensors, 0.0
            )  # (N, num_bins)
            sensors_color = jnp.max(sensors_color, axis=0)  # (num_bins,)
            return sensors_color

        color_ids = jnp.arange(len(self.propositions), dtype=jnp.int32)
        lidar = jax.vmap(compute_color_lidar)(color_ids)  # (C, num_bins)
        return lidar

    def _compute_rgb_lidar(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Computes a single lidar with RGB and intensity of the most intense zone.

        Returns: (num_bins, 5) array (r, g, b, intensity, detected)
        """
        pos = state.position
        bin_size = 2.0 * jnp.pi / params.num_lidar_bins
        heading = jnp.array([jnp.cos(state.angle), jnp.sin(state.angle)])

        centers = state.zone_centers
        rgbs = self.rgbs[state.zone_colors]

        direction = centers - pos
        dist = jnp.linalg.norm(direction, axis=1)
        intensity = jnp.exp(-params.exp_gain * dist)

        direction_norm = direction / (dist[:, None] + _EPS)
        dotp = jnp.dot(direction_norm, heading)
        cross = jnp.cross(direction_norm, heading)
        angle = jnp.arctan2(cross, dotp) % (2.0 * jnp.pi)
        bin_idx = jnp.floor(angle / bin_size).astype(jnp.int32)

        # For each bin, find the zone with max intensity
        def max_intensity_in_bin(i):
            mask = bin_idx == i
            intensities_in_bin = jnp.where(mask, intensity, -1.0)
            max_idx = jnp.argmax(intensities_in_bin)
            max_intensity = intensity[max_idx]
            rgb = rgbs[max_idx]
            # If not detected, return zeros, otherwise return the values
            detected = jnp.any(mask)
            # Normalize rgb and intensity to [-1, 1]
            normalized_rgb = rgb * 2.0 - 1.0
            normalized_intensity = max_intensity * 2.0 - 1.0
            return jnp.where(
                detected,
                jnp.concatenate(
                    [normalized_rgb, jnp.array([normalized_intensity, 1.0])]
                ),
                jnp.zeros(5),
            )

        bins = jnp.arange(params.num_lidar_bins)
        return jax.vmap(max_intensity_in_bin)(bins)

    def _compute_2_rgb_lidar(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Computes a single lidar with RGB and intensity of the most intense zone.

        Returns: (num_bins, num_layers, 5) array (r, g, b, intensity, detected)
        """
        num_layers = 2

        pos = state.position
        bin_size = 2.0 * jnp.pi / params.num_lidar_bins
        heading = jnp.array([jnp.cos(state.angle), jnp.sin(state.angle)])

        centers = state.zone_centers
        rgbs = self.rgbs[state.zone_colors].astype(jnp.float32) / 255.0  # (N,3)

        direction = centers - pos
        dist = jnp.linalg.norm(direction, axis=1)
        intensity = jnp.exp(-params.exp_gain * dist)

        direction_norm = direction / (dist[:, None] + _EPS)
        dotp = jnp.dot(direction_norm, heading)
        cross = jnp.cross(direction_norm, heading)
        angle = jnp.arctan2(cross, dotp) % (2.0 * jnp.pi)
        bin_idx = jnp.floor(angle / bin_size).astype(jnp.int32)

        # For each bin, find the top K zones sorted by intensity (closest first)
        def layers_in_bin(i):
            mask = bin_idx == i

            # We mask intensities of objects not in this bin to -1.0
            # (Since real intensity is exp(-dist), it is always > 0)
            intensities_in_bin = jnp.where(mask, intensity, -1.0)

            # Use lax.top_k to get the indices of the highest intensities efficiently
            # This automatically handles sorting by intensity (closest objects first)
            top_intensities, top_indices = jax.lax.top_k(intensities_in_bin, num_layers)

            # Gather data for the top K objects
            top_rgbs = rgbs[top_indices]  # (num_layers, 3)

            # Check if the slot actually contains a valid object (intensity > -1.0)
            # We use a slightly higher threshold than -1.0 to be safe against float errors
            detected_mask = top_intensities > -0.5

            # Prepare the combined features: RGB (3) + Intensity (1) + Detected (1)
            # Shape: (num_layers, 5)
            layer_features = jnp.concatenate(
                [
                    top_rgbs,
                    top_intensities[:, None],
                    detected_mask[:, None].astype(jnp.float32),
                ],
                axis=1,
            )

            # Zero out features for empty slots (where detected is False)
            return jnp.where(
                detected_mask[:, None], layer_features, jnp.zeros_like(layer_features)
            )

        bins = jnp.arange(params.num_lidar_bins)

        # Output shape: (num_bins, num_layers, 5)
        return jax.vmap(layers_in_bin)(bins)

    def _compute_zone_vecs(self, state: EnvState, params: EnvParams) -> jax.Array:
        pos = state.position
        heading = jnp.array([jnp.cos(state.angle), jnp.sin(state.angle)])
        centers = state.zone_centers
        directions = centers - pos  # (N,2)
        dists = jnp.linalg.norm(directions, axis=1)  # (N,)
        intensity = jnp.exp(-params.exp_gain * dists)
        direction_norm = directions / (dists[:, None] + _EPS)
        dotp = jnp.dot(direction_norm, heading)
        cross = jnp.cross(direction_norm, heading)
        angle = jnp.arctan2(cross, dotp)
        sin_angle = jnp.sin(angle)
        cos_angle = jnp.cos(angle)
        zone_vecs = jnp.stack([intensity, sin_angle, cos_angle], axis=1)  # (N,3)
        return zone_vecs

    @override
    def compute_propositions(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Compute which zones the agent is currently inside.

        Returns an int32 array of shape (C,) containing the color ids of the zones
        the agent is inside, with -1 as padding.
        """
        pos = state.position  # (2,)
        centers = state.zone_centers  # (N,2)
        colors = state.zone_colors  # (N,)

        dists = jnp.linalg.norm(centers - pos, axis=1)  # (N,)
        inside = dists < params.zone_radius  # (N,)

        def compute_color_prop(color_id: jax.Array) -> jax.Array:
            mask_color = colors == color_id  # (N,)
            inside_color = jnp.logical_and(mask_color, inside)  # (N,)
            return jax.lax.cond(jnp.any(inside_color), lambda: color_id, lambda: -1)

        color_ids = jnp.arange(len(self.propositions), dtype=jnp.int32)
        propositions = jax.vmap(compute_color_prop)(color_ids)  # (C,)
        return jnp.sort(propositions, descending=True)

    @property
    @override
    def assignments(self) -> list[Assignment]:
        """Returns all possible assignments in the environment."""
        assignments = [Assignment(frozenset({color})) for color in self.propositions]
        assignments.append(Assignment(frozenset()))  # empty assignment
        return assignments

    @override
    def get_renderer(
        self, env_params: EnvParams, **kwargs
    ) -> "BaseRenderer[ObsFeatures, ResetOptions]":
        """Returns a renderer for the environment."""
        from .renderer import Renderer

        return Renderer(env_params, **kwargs)

    @override
    def plot_trajectories(
        self,
        trajs: EnvState,
        lengths: jax.Array,
        params: EnvParams,
        **plotting_kwargs,
    ) -> None:
        """Plots trajectories of environment states.

        Args:
            trajs: Batched EnvStates of shape (num_episodes, max_length, ...)
            params: Environment parameters
            plotting_kwargs: Additional keyword arguments for the plotting function
        """
        zone_positions = trajs.zone_centers[:, 0].tolist()
        zone_colors = trajs.zone_colors[:, 0].tolist()
        zone_colors = [[self.propositions[i] for i in cs] for cs in zone_colors]
        paths = [
            trajs.position[i, : lengths[i]].tolist() for i in range(lengths.shape[0])
        ]
        draw_trajectories(zone_positions, zone_colors, paths, **plotting_kwargs)
