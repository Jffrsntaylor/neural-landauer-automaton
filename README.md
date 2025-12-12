![Simulation preview](assets/simulation_output.gif)

# Neural Landauer Automaton — Simulating Vacuum Phase Transitions with Neural Cellular Automata

Simulates a Landauer-inspired vacuum that quenches from noisy chaos into a Fibonacci-like ordered phase while primordial black holes (PBHs) melt local domain walls.

## Emergence & Results
![Entropy trajectory](assets/entropy_history.png)

The vertical drop in loss represents the **Quench**—a rapid phase transition from high-entropy chaos to a stable Fibonacci ground state. After the drop, the automaton holds the ordered texture while respecting PBH-induced disorder.

## How It Works
- **Perception**: Sobel gradients with wrap padding (toroidal topology) feed the rule.
- **Update rule**: 1x1 mix → sin/ReLU → 1x1 output, stochastic firing (~50%), damping for stability.
- **Training objective**: Texture (Gram) loss + moment (OT-lite) loss + weight penalty to learn the ordered phase from random noise.
- **PBHs**: Clamped -1.0 pixels act as high-entropy defects; surrounding domain walls “melt.”

## Project Layout
- `model.py` — NCA core with toroidal perception and PBH masking.
- `train.py` — Training loop, entropy plot export (entropy_history.png), checkpoints in `checkpoints/`.
- `simulate.py` — PBH experiment and GIF export (simulation_output.gif).
- `assets/` — Curated outputs kept under version control.
- `requirements.txt` — Pinned JAX/Flax/Optax stack.
- `LICENSE` — MIT.

## Quick Start
1) Environment
```bash
python -m venv .venv
.venv/Scripts/activate  # Windows
# or: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt  # numpy pinned <2 for ABI safety
```

2) Train (CPU on native Windows; GPU via WSL2+CUDA)
```bash
python train.py --height 48 --width 48 --channels 16 --batch 4 --rollout 32 --steps 500 --lr 1e-3
```
Checkpoints land in `checkpoints/landauer_params_*`.

3) One-liner demo (uses latest checkpoint or random weights)
```bash
python simulate.py --checkpoint checkpoints --steps 200 --height 64 --width 64 --channels 16 --black-hole 24,24 --black-hole 40,12 --radius 4 --output assets/simulation_output.gif
```

## Notes & Tips
- Increase `--fire-rate` toward 1.0 for synchronous updates; lower for more stochastic dynamics.
- If patterns stagnate, gently raise `damping` in `model.py`; if unstable, lower it.
- First run JIT-compiles; expect a short pause.
- GPU on Windows: use WSL2 with CUDA and install the Linux CUDA JAX wheels; native Windows wheels are CPU-only.

## License
MIT License (see `LICENSE`).
