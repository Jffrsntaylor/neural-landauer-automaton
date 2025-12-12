
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import imageio
import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints

import os_config
from model import NeuralLandauerAutomaton, apply_black_hole_mask

os_config.setup_gpu()

Array = jnp.ndarray


def parse_args():
    """CLI for PBH-melted domain wall simulation and GIF export."""
    parser = argparse.ArgumentParser(description="Simulate PBH-melted domain walls in the Landauer NCA")
    parser.add_argument("--checkpoint", type=str, default="checkpoints", help="Directory with landauer_params_ checkpoint")
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--fire-rate", type=float, default=0.5)
    parser.add_argument(
        "--black-hole",
        action="append",
        default=[],
        help="Comma-separated x,y for PBH centers (repeatable)",
    )
    parser.add_argument("--radius", type=int, default=3, help="Radius of the melted PBH core (pixels)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="simulation_output.gif")
    return parser.parse_args()


def parse_coords(coord_strings: Iterable[str], height: int, width: int) -> List[Tuple[int, int]]:
    """Parse and clamp PBH coordinate strings into grid-safe tuples."""
    coords = []
    for item in coord_strings:
        if item is None:
            continue
        parts = item.split(",")
        if len(parts) != 2:
            raise ValueError(f"Bad black-hole coordinate: {item}")
        x, y = int(parts[0]), int(parts[1])
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        coords.append((x, y))
    if not coords:
        coords.append((width // 2, height // 2))
    return coords


def make_pbh_mask(height: int, width: int, coords: List[Tuple[int, int]], radius: int) -> Array:
    """Build a binary mask marking PBH cores (with channel axis)."""
    yy, xx = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij")
    mask = jnp.zeros((height, width, 1), dtype=jnp.float32)
    for x, y in coords:
        dist = jnp.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        mask = jnp.maximum(mask, (dist <= radius)[..., None].astype(jnp.float32))
    return mask


def render_frame(state: Array, pbh_mask: Array) -> np.ndarray:
    """Convert state tensor and PBH overlay into a uint8 RGB frame."""
    data = np.array(state)
    if data.shape[-1] < 3:
        data = np.repeat(data, 3, axis=-1)
    if data.shape[-1] > 3:
        data = data[..., :3]

    # Normalize for visualization.
    data = (data - data.min()) / (data.max() - data.min() + 1e-6)

    mask = np.array(pbh_mask)[..., 0]
    overlay = np.stack([mask, np.zeros_like(mask), np.zeros_like(mask)], axis=-1)
    blend = np.clip(data * 0.8 + overlay * 0.2, 0.0, 1.0)
    return (blend * 255).astype(np.uint8)


def run(args):
    """Main simulation loop: load params, apply PBHs, evolve, and write GIF."""
    rng = jax.random.PRNGKey(args.seed)
    model = NeuralLandauerAutomaton(channels=args.channels)
    dummy = jnp.zeros((1, args.height, args.width, args.channels), dtype=jnp.float32)
    params = model.init(rng, dummy, rng)["params"]

    ckpt_dir = Path(args.checkpoint).resolve()
    restored = checkpoints.restore_checkpoint(ckpt_dir, target=None, prefix="landauer_params_")
    if restored is not None:
        params = restored
    else:
        print("Warning: checkpoint not found, using randomly initialized parameters.")

    coords = parse_coords(args.black_hole, args.height, args.width)
    pbh_mask = make_pbh_mask(args.height, args.width, coords, args.radius)
    pbh_mask = jnp.expand_dims(pbh_mask, axis=0)

    state = jax.random.uniform(rng, (1, args.height, args.width, args.channels), minval=-1.0, maxval=1.0)
    state = apply_black_hole_mask(state, pbh_mask)

    frames = []
    for _ in range(args.steps):
        rng, sub = jax.random.split(rng)
        state = model.apply({"params": params}, state, rng=sub, pbh_mask=pbh_mask, fire_rate=args.fire_rate)
        state = apply_black_hole_mask(state, pbh_mask)
        frames.append(render_frame(state[0], pbh_mask[0]))

    imageio.mimsave(args.output, frames, fps=12)
    print(f"Saved simulation to {args.output}")


if __name__ == "__main__":
    run(parse_args())
