
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import jax
import jax.numpy as jnp
import optax
from flax import config as flax_config
from flax.training import checkpoints, train_state
from tqdm import trange

import os_config
from model import NeuralLandauerAutomaton, sense_field

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

os_config.setup_gpu()
flax_config.update("flax_use_orbax_checkpointing", False)

Array = jnp.ndarray


class AutomatonState(train_state.TrainState):
    pass


def fibonacci_phase(height: int, width: int, channels: int) -> Array:
    """Generate a quasiperiodic Fibonacci-like target texture across channels."""
    phi = (1.0 + jnp.sqrt(5.0)) / 2.0
    xs = jnp.linspace(-jnp.pi, jnp.pi, width)
    ys = jnp.linspace(-jnp.pi, jnp.pi, height)
    grid_x, grid_y = jnp.meshgrid(xs, ys)

    waves = [
        jnp.sin(grid_x + phi * grid_y),
        jnp.sin(phi * grid_x - grid_y),
        jnp.sin(phi**2 * grid_x + phi * grid_y),
    ]
    texture = sum(waves) / len(waves)
    texture += 0.25 * jnp.sin(3.0 * grid_x) * jnp.cos(2.0 * grid_y)
    texture = texture - texture.min()
    texture = 2.0 * texture / (texture.max() + 1e-6) - 1.0

    channels_list = []
    for idx in range(channels):
        freq = 1.0 + 0.15 * idx
        channels_list.append(jnp.sin(freq * texture))
    target = jnp.stack(channels_list, axis=-1)
    return target


def gram_matrix(features: Array) -> Array:
    """Compute channel correlation matrix to measure texture similarity."""
    b, h, w, c = features.shape
    features = features.reshape(b, h * w, c)
    # Physicist Notes: Gram matrix captures correlation of field channels -> texture energy landscape.
    gram = jnp.einsum("bij,bik->bjk", features, features) / (h * w * c)
    return gram


def texture_loss(pred: Array, target: Array) -> Array:
    """Style/texture loss between predicted and target textures."""
    pred_features = sense_field(pred)
    target_features = sense_field(target)
    g_pred = gram_matrix(pred_features)
    g_target = gram_matrix(target_features)
    return jnp.mean((g_pred - g_target) ** 2)


def moment_transport_loss(pred: Array, target: Array) -> Array:
    """Low-order optimal transport proxy matching channel means/variances."""
    # Physicist Notes: matching first/second moments approximates a low-order optimal transport cost.
    mean_loss = jnp.mean((jnp.mean(pred, axis=(1, 2)) - jnp.mean(target, axis=(1, 2))) ** 2)
    var_loss = jnp.mean((jnp.var(pred, axis=(1, 2)) - jnp.var(target, axis=(1, 2))) ** 2)
    return mean_loss + var_loss


def create_train_state(rng, height: int, width: int, channels: int, lr: float):
    """Initialize model parameters and optimizer state."""
    model = NeuralLandauerAutomaton(channels=channels)
    dummy = jnp.zeros((1, height, width, channels), dtype=jnp.float32)
    params = model.init(rng, dummy, rng)["params"]
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    return AutomatonState.create(apply_fn=model.apply, params=params, tx=tx), model


def rollout(model, params, init_state, rng, steps: int, fire_rate: float):
    """Unroll the automaton for a fixed number of steps, returning trajectory."""
    def step_fn(carry, _):
        state, key = carry
        key, sub = jax.random.split(key)
        state = model.apply({"params": params}, state, rng=sub, fire_rate=fire_rate)
        return (state, key), state

    (final_state, _), trajectory = jax.lax.scan(step_fn, (init_state, rng), None, length=steps)
    return final_state, trajectory


def train_step(state: AutomatonState, rng, target: Array, rollout_steps: int, fire_rate: float):
    """Single optimization step: sample noise, evolve, compute loss, apply grads."""
    batch_size, height, width, channels = target.shape
    model = NeuralLandauerAutomaton(channels=channels)

    def loss_fn(params):
        key, noise_key = jax.random.split(rng)
        init_state = jax.random.uniform(
            noise_key, (batch_size, height, width, channels), minval=-1.0, maxval=1.0
        )
        final_state, _ = rollout(model, params, init_state, key, steps=rollout_steps, fire_rate=fire_rate)
        tex = texture_loss(final_state, target)
        ot = moment_transport_loss(final_state, target)
        weight_penalty = 1e-4 * sum(jnp.mean(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
        return tex + 0.1 * ot + weight_penalty

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train Neural Landauer Automaton")
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--rollout", type=int, default=24)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fire-rate", type=float, default=0.5)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()
    rng = jax.random.PRNGKey(0)

    target_single = fibonacci_phase(args.height, args.width, args.channels)
    target = jnp.broadcast_to(target_single, (args.batch, args.height, args.width, args.channels))

    state, model = create_train_state(rng, args.height, args.width, args.channels, args.lr)
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    loss_history = []
    progress = trange(args.steps, desc="Training")
    for step in progress:
        rng, step_rng = jax.random.split(rng)
        state, loss = train_step(state, step_rng, target, args.rollout, args.fire_rate)
        loss_history.append(float(loss))
        if step % 10 == 0:
            progress.set_postfix(loss=float(loss))

    checkpoints.save_checkpoint(
        checkpoint_dir,
        target=state.params,
        step=args.steps,
        prefix="landauer_params_",
        overwrite=True,
    )
    plt.figure(figsize=(6, 4))
    plt.plot(loss_history, label="loss")
    plt.xlabel("Step")
    plt.ylabel("Loss (entropy/order proxy)")
    plt.title("Entropy/Order trajectory")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("entropy_history.png")
    print(f"Saved checkpoint to {checkpoint_dir}")


if __name__ == "__main__":
    main()
