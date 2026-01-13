# SRLP — Self-Regulating Learning Pressure

SRLP is a training-time control algorithm that dynamically adjusts gradient pressure
based on short-term loss volatility to stabilize neural network learning.

## Motivation
Modern neural networks use fixed learning behavior during training.
SRLP introduces a feedback mechanism that adapts learning pressure based on training dynamics.

## Algorithm Overview
- Monitors recent loss volatility
- Reduces learning pressure during unstable phases
- Gradually restores learning pressure as training stabilizes

## Current Version
- SRLP v0 / v1: volatility-based pressure control

## Experiments
- Dataset: MNIST
- Model: Simple MLP
- Framework: PyTorch

## Files
- `baseline.py` — standard training without SRLP
- `srlp_v0.py` — training with SRLP logic

## Status
This project is under active research and development.

## Future Work
- Optimizer-level integration
- Per-layer pressure control
- CNN and Transformer experiments
- Research publication
