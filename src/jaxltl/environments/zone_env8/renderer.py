"""A 2D renderer for the zone environment based on pygame."""

from functools import partial
from typing import override

import jax
import jax.numpy as jnp
import pygame
from pygame import gfxdraw

from jaxltl.environments.renderer.renderer import ContinuousTimeRenderer
from jaxltl.environments.zone_env8.zone_env8 import (
    EnvParams,
    EnvState,
    ObsFeatures,
    ResetOptions,
)


class Renderer(ContinuousTimeRenderer[ObsFeatures, ResetOptions]):
    def __init__(
        self,
        params: EnvParams,
        screen_size: int = 800,
        grid_size: int = 50,
    ):
        super().__init__("Zone Environment", screen_size)

        self._params = params
        self._screen_size = screen_size

        self._background = pygame.Surface(self._screen.get_size())

        self._world_to_screen_scale = screen_size / params.world_size
        self._agent_radius_px = int(0.1 * self._world_to_screen_scale)
        self._zone_radius_px = int(params.zone_radius * self._world_to_screen_scale)

        # Checkerboard background
        self.grid_size = grid_size
        self._grid_color_1 = (248, 250, 252)
        self._grid_color_2 = (241, 245, 249)
        self._render_background()

        # Agent color
        self._agent_color = (59, 130, 246)  # blue-500
        self._agent_heading_color = (59, 130, 246, 180)  # blue-500 with alpha

        # Color mapping for zones
        self._zone_colors = dict(
            enumerate(
                [
                    (31, 119, 180),
                    (255, 127, 14),
                    (44, 160, 44),
                    (214, 39, 40),
                    (148, 103, 189),
                    (140, 86, 75),
                    (227, 119, 194),
                    (127, 127, 127),
                ]
            )
        )

    def _render_background(self):
        """Draw checkerboard background."""
        self._background.fill(self._grid_color_1)
        for y in range(0, self._screen_size, self.grid_size):
            for x in range(0, self._screen_size, self.grid_size):
                if (y // self.grid_size + x // self.grid_size) % 2 == 1:
                    rect = pygame.Rect(x, y, self.grid_size, self.grid_size)
                    self._background.fill(self._grid_color_2, rect)

    def _render_zones(self, state: EnvState):
        centers = self._world_to_screen(state.zone_centers).tolist()
        for i, center in enumerate(centers):
            color_id = int(state.zone_colors[i])
            col = self._zone_colors.get(color_id, (0, 0, 0))
            self._draw_circle(self._screen, col, center, self._zone_radius_px)

    def _draw_circle(self, surface, color, position, radius):
        """Draw an anti-aliased filled circle."""
        gfxdraw.aacircle(surface, position[0], position[1], radius, color)
        gfxdraw.filled_circle(surface, position[0], position[1], radius, color)

    @partial(jax.jit, static_argnames=("self",))
    def _world_to_screen(self, pos: jax.Array) -> jax.Array:
        """Convert world coordinates to screen coordinates."""
        pos = (pos + self._params.world_size / 2) * self._world_to_screen_scale
        pos = pos.at[:, 1].set(self._screen_size - pos[:, 1])
        return pos.astype(jnp.int32)

    @override
    def render_with_interpolation(
        self,
        state: EnvState,
        previous_state: EnvState,
        obs: ObsFeatures | None,
        alpha: float,
    ):
        """Render the environment state."""
        self._screen.blit(self._background, (0, 0))
        self._render_zones(state)

        # Interpolation for smooth rendering
        interpolated_position = (
            previous_state.position * (1.0 - alpha) + state.position * alpha
        )
        angle_diff = (state.angle - previous_state.angle + jnp.pi) % (
            2 * jnp.pi
        ) - jnp.pi
        interpolated_angle = previous_state.angle + alpha * angle_diff

        self._draw_agent(interpolated_position, interpolated_angle)

        pygame.display.flip()

    @override
    def _format_obs(self, obs: ObsFeatures) -> str:
        """Neatly formats the observations and propositions into a single string."""
        output = []
        output.append(f"Type: {type(obs).__name__}\n")
        for field, value in obs._asdict().items():
            if not isinstance(value, jax.Array):
                output.append(f"  {field}: {value}\n")
                continue
            if value.ndim == 2:
                output.append(f"  {field}: shape {value.shape}\n")
                if field == "rgb_lidar":
                    output.append(self._format_rgb_lidar_field(value))
                elif field == "lidar":
                    output.append(self._format_lidar_field(value))
                elif field == "zone_vecs":
                    output.append(self._format_privileged_field(value))
            else:
                with jnp.printoptions(precision=2, suppress=True):
                    output.append(f"  {field}: {value}\n")
        return "".join(output)

    def _format_lidar_field(self, value: jax.Array) -> str:
        lines = []
        num_colors, num_bins = value.shape

        # Header
        header_parts = [f"{'Bin':>3}"]
        header_parts.extend([f"{f'C{i}':>5}" for i in range(num_colors)])
        lines.append(f"    {' | '.join(header_parts)}\n")

        # Separator
        separator_parts = [f"{'-' * 3}"]
        separator_parts.extend([f"{'-' * 5}" for _ in range(num_colors)])
        lines.append(f"    {'-+-'.join(separator_parts)}\n")

        # Data rows
        for i in range(num_bins):
            row_parts = [f"{i:3d}"]
            row_parts.extend([f"{value[j, i]:5.2f}" for j in range(num_colors)])
            lines.append(f"    {' | '.join(row_parts)}\n")

        return "".join(lines)

    def _format_rgb_lidar_field(self, value: jax.Array) -> str:
        lines = []
        detected_mask = value[:, 4] > 0
        detected_count = jnp.sum(detected_mask)
        lines.append(f"    Detected Lidar Rays: {detected_count}/{value.shape[0]}\n")

        # Header
        header = f"      {'Bin':>3} | {'R':>5} | {'G':>5} | {'B':>5} | {'Intensity':>9} | {'Detected':>8}\n"
        lines.append(header)
        lines.append(
            f"      {'-' * 3}-+-{'-' * 5}-+-{'-' * 5}-+-{'-' * 5}-+-{'-' * 9}-+-{'-' * 8}\n"
        )

        for i, row in enumerate(value):
            lines.append(
                f"      {i:3d} | {row[0]:5.2f} | {row[1]:5.2f} | {row[2]:5.2f} | {row[3]:9.2f} | {int(row[4]):8d}\n"
            )
        return "".join(lines)

    def _format_privileged_field(self, value: jax.Array) -> str:
        lines = []
        num_zones, _ = value.shape
        lines.append(f"    Privileged info for {num_zones} zones:\n")

        # Header
        header = f"      {'Zone':>4} | {'Intensity':>5} | {'Sin':>5} | {'Cos':>5}\n"
        lines.append(header)
        lines.append(f"      {'-' * 4}-+-{'-' * 9}-+-{'-' * 5}-+-{'-' * 5}\n")

        # Data rows
        for i, row in enumerate(value):
            lines.append(
                f"      {i:4d} | {row[0]:5.2f} | {row[1]:5.2f} | {row[2]:5.2f}\n"
            )

        return "".join(lines)

    def _draw_agent(self, position: jax.Array, angle: jax.Array):
        # Draw agent heading as a rectangle
        cos_angle = jnp.cos(angle)
        sin_angle = jnp.sin(angle)
        rect_w = 0.02
        rect_l = 0.2
        corners = jnp.array(
            [
                [0, -rect_w / 2],
                [rect_l, -rect_w / 2],
                [rect_l, rect_w / 2],
                [0, rect_w / 2],
            ]
        )

        # Rotate and translate corners
        rotation_matrix = jnp.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
        rotated_corners = jnp.dot(corners, rotation_matrix.T)
        translated_corners = rotated_corners + position

        agent_and_corners = jnp.vstack([position, translated_corners])
        screen_positions = self._world_to_screen(agent_and_corners).tolist()
        agent_pos = screen_positions[0]
        self._draw_circle(
            self._screen, self._agent_color, agent_pos, self._agent_radius_px
        )
        corners = screen_positions[1:]
        gfxdraw.filled_polygon(self._screen, corners, self._agent_heading_color)
        gfxdraw.aapolygon(self._screen, corners, self._agent_heading_color)

    @override
    def get_action(self, keys: pygame.key.ScancodeWrapper) -> jax.Array:
        """Gets an action from user input."""

        force = 0.0
        angular_velocity = 0.0

        if keys[pygame.K_w]:
            force = 1.0
        if keys[pygame.K_s]:
            force = -1.0
        if keys[pygame.K_a]:
            angular_velocity = 1.0
        if keys[pygame.K_d]:
            angular_velocity = -1.0

        return jnp.array([force, angular_velocity])
