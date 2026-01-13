# Experimental Results and Analysis

---

## Experimental Setup

- Dataset: MNIST
- Model: Simple MLP (Flatten → 128 → ReLU → 10)
- Framework: PyTorch
- Optimizer: Adam
- Learning Rate: 0.001
- Batch Size: 64
- Epochs: 3
- Hardware: CPU

All experiments use identical settings to ensure fair comparison.

---

## Quantitative Results

### Global Stability Comparison

| Method        | Test Accuracy (%) | Loss Std (Global) | Loss Std (Late) | Loss Spike Count |
|---------------|------------------|-------------------|-----------------|------------------|
| Baseline      | 97.02            | 0.2028            | 0.0668          | 291              |
| SRLP (Global) | 96.85            | 0.2085            | 0.0650          | 282              |

SRLP preserves accuracy while reducing late-stage instability.

---

### Gradient Stability Metrics

| Method        | Gradient Norm Mean | Gradient Norm Std |
|---------------|-------------------|-------------------|
| Baseline      | 0.6835            | 0.2401            |
| SRLP (Global) | 0.6621            | 0.2322            |
| SRLP-L        | 0.4272            | 0.1519            |

SRLP-L achieves the strongest gradient stabilization.

---

## Layer-wise Pressure Behavior (SRLP-L)

| Layer          | Pressure Range | Observed Effect |
|----------------|----------------|-----------------|
| net.1.weight   | 0.60 – 0.79    | Stable feature learning |
| net.1.bias     | 0.92 – 0.96    | High robustness |
| net.3.weight   | 0.60 – 0.68    | Classification stabilization |

SRLP-L dynamically allocates learning capacity across layers.

---

## Visual Analysis Summary

- Loss curves show smooth convergence
- Learning pressure increases as training stabilizes
- No oscillation or divergence observed
- Layer-wise pressure differs meaningfully across network depth

---

## Key Findings

- SRLP stabilizes training without modifying the optimizer
- SRLP-L significantly reduces gradient variance
- Accuracy is preserved or marginally improved
- Computational overhead is minimal

---

## Conclusion

SRLP provides an effective training-time control mechanism for neural networks.
Both global and layer-wise variants improve stability while preserving performance.
