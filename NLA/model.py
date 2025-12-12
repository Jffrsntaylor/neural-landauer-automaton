
from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

Array = jnp.ndarray


def _make_depthwise_kernel(base_kernel: Array, channels: int) -> Array:
    """Construct a depthwise convolution kernel repeated for each channel."""
    base = base_kernel.astype(jnp.float32)
    # Make (H, W, in_channels=1, out_channels=channels) for depthwise grouping.
    return jnp.repeat(base[..., None, None], channels, axis=3)


def sobel_kernels(channels: int) -> Tuple[Array, Array, Array]:
    """Return identity and Sobel gradient kernels for perception."""
    sobel_x = jnp.array(
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=jnp.float32
    ) / 4.0
    sobel_y = jnp.array(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=jnp.float32
    ) / 4.0
    identity = jnp.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=jnp.float32)
    return (
        _make_depthwise_kernel(identity, channels),
        _make_depthwise_kernel(sobel_x, channels),
        _make_depthwise_kernel(sobel_y, channels),
    )


def depthwise_conv(x: Array, kernel: Array) -> Array:
    """Apply wrap-padded depthwise convolution to emulate toroidal topology."""
    # Toroidal topology: wrap pad so opposite edges interact; then run valid conv.
    x_padded = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="wrap")
    return jax.lax.conv_general_dilated(
        lhs=x_padded,
        rhs=kernel,
        window_strides=(1, 1),
        padding="VALID",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        feature_group_count=x.shape[-1],
    )


def sense_field(state: Array) -> Array:
    """Expose local field values and gradients as perception channels."""
    identity, kx, ky = sobel_kernels(state.shape[-1])
    base = depthwise_conv(state, identity)
    grad_x = depthwise_conv(state, kx)
    grad_y = depthwise_conv(state, ky)
    # Physicist Notes: concatenate local field and its spatial gradients to expose flux and curvature.
    return jnp.concatenate([base, grad_x, grad_y], axis=-1)


class NeuralLandauerAutomaton(nn.Module):
    """
    Neural CA that learns a Landauer vacuum-like phase transition.

    channels: number of field channels (information density slices).
    hidden_channels: latent width of the mixing layer.
    fire_rate: fraction of cells updated per step (stochastic CA mask).
    damping: scales updates to keep dynamics in a stable basin.
    """

    channels: int
    hidden_channels: int = 96
    fire_rate: float = 0.5
    use_sin_activation: bool = True
    damping: float = 0.25

    @nn.compact
    def __call__(
        self,
        state: Array,
        rng: jax.random.KeyArray,
        pbh_mask: Optional[Array] = None,
        fire_rate: Optional[float] = None,
    ) -> Array:
        fire = fire_rate if fire_rate is not None else self.fire_rate

        perception = sense_field(state)
        mix = nn.Conv(
            features=self.hidden_channels,
            kernel_size=(1, 1),
            kernel_init=nn.initializers.variance_scaling(
                0.1, "fan_avg", "truncated_normal"
            ),
            name="mix_conv",
        )(perception)
        # Physicist Notes: nonlinearity acts like a local equation of state; sin introduces oscillatory stiffness.
        if self.use_sin_activation:
            activated = jnp.sin(mix)
        else:
            activated = nn.relu(mix)

        delta = nn.Conv(
            features=self.channels,
            kernel_size=(1, 1),
            kernel_init=nn.initializers.normal(0.02),
            name="update_conv",
        )(activated)

        update_mask = (
            jax.random.uniform(rng, state.shape[:-1] + (1,)) <= fire
        ).astype(jnp.float32)
        # Physicist Notes: stochastic mask ~ thermal kicks; only a fraction of cells evolve each step.
        state_update = delta * update_mask * self.damping
        new_state = state + state_update

        if pbh_mask is not None:
            # Physicist Notes: clamp PBH pixels to maximal entropy, acting as absorbing defects.
            new_state = jnp.where(pbh_mask, -jnp.ones_like(new_state), new_state)
        return new_state


def apply_black_hole_mask(state: Array, mask: Array) -> Array:
    """Clamp masked pixels to -1 to represent PBH-induced disorder."""
    return jnp.where(mask, -jnp.ones_like(state), state)


__all__ = [
    "NeuralLandauerAutomaton",
    "sense_field",
    "sobel_kernels",
    "depthwise_conv",
    "apply_black_hole_mask",
]
