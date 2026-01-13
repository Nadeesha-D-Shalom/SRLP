# SRLP — Self-Regulating Learning Pressure

SRLP (Self-Regulating Learning Pressure) is a training-time control algorithm that
dynamically adjusts gradient update strength based on short-term loss volatility
to stabilize neural network learning without sacrificing performance.

---

## Motivation

Most neural network training pipelines rely on fixed learning dynamics determined
by static optimizers and learning rates. However, training instability often varies
across time, especially during early or noisy phases.

SRLP introduces a **feedback-driven learning control mechanism** that adapts
gradient pressure in response to observed loss volatility, enabling safer and
more interpretable training behavior.

---

## Algorithm Overview

SRLP operates during training and performs the following steps:

- Monitors short-term loss volatility using a sliding window
- Computes a continuous learning pressure signal based on volatility
- Reduces gradient magnitude during unstable phases
- Gradually restores full learning pressure as training stabilizes

The algorithm is **model-agnostic**, **optimizer-independent**, and requires
no modification to the network architecture.

---

## Current Version

- **SRLP v1.1** — Continuous, tuned learning-pressure control  
  - Smooth pressure adaptation  
  - No hard thresholds  
  - Preserves convergence and accuracy  

---

## Repository Structure

```
SRLP/
├── srlp.py
├── experiments/
│   ├── baseline.py
│   └── srlp_mnist.py
├── data/
├── README.md
├── requirements.txt
├── .gitignore
```

### File Descriptions

- `srlp.py` — Core SRLP algorithm (learning-pressure controller)
- `experiments/baseline.py` — Standard MNIST training without SRLP
- `experiments/srlp_mnist.py` — MNIST training with SRLP enabled

---

## Experiments

- Dataset: MNIST
- Model: Simple MLP
- Framework: PyTorch
- Evaluation focus:
  - Training stability
  - Loss volatility
  - Learning dynamics
  - Final accuracy preservation

---

## Status

This project is under **active research and development**.
The current focus is on stability analysis, metric refinement, and
engineering-grade algorithm design.

---

## Future Work

- Advanced stability metrics (loss spike count, gradient variance)
- Optimizer-level integration
- Per-layer and adaptive pressure control
- CNN and Transformer experiments
- Research paper and public release

---

## Author

Developed and maintained by **Nadeesha D Shalom**  
Research focus: Learning dynamics, training stability, and AI control mechanisms.
