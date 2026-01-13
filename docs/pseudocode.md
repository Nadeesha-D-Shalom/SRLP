# Pseudocode and Complexity Analysis

---

## Algorithm 1: SRLP (Global Self-Regulating Learning Pressure)

### Inputs
- Training dataset D
- Neural network model with parameters θ
- Optimizer OPT (e.g., SGD, Adam)
- Loss function L(·)
- Window size W
- Sensitivity constant k
- Minimum pressure p_min

### Outputs
- Trained model parameters θ

---

### Pseudocode

Initialize loss buffer B as empty queue with capacity W

for each training step t do
    Sample minibatch (x_t, y_t) from D
    Compute loss:
        L_t = L(model(x_t), y_t)

    Append L_t to buffer B

    if size(B) == W then
        Compute loss volatility:
            σ_t = StdDev(B)

        Compute learning pressure:
            p_t = clip( 1 / (1 + k · σ_t), p_min, 1.0 )
    else
        p_t = 1.0
    end if

    Compute gradients:
        g_t = ∇_θ L_t

    Apply pressure:
        g_t ← p_t · g_t

    Update parameters:
        θ ← OPT(θ, g_t)
end for

Return trained parameters θ

---

### Computational Complexity

- Time per step:
    O(1) for pressure computation (windowed statistics)
- Space:
    O(W) for loss buffer

SRLP introduces negligible overhead relative to standard training.

## Algorithm 2: SRLP-L (Layer-wise Self-Regulating Learning Pressure)

### Inputs
- Training dataset D
- Neural network with layers ℓ = 1 … L
- Optimizer OPT
- Loss function L(·)
- Window size W
- Sensitivity constant k
- Layer-specific minimum pressures p_min^ℓ

### Outputs
- Trained model parameters θ

---

### Pseudocode

For each layer ℓ:
    Initialize gradient norm buffer B^ℓ with capacity W

for each training step t do
    Sample minibatch (x_t, y_t) from D
    Compute loss:
        L_t = L(model(x_t), y_t)

    Compute gradients:
        g_t^ℓ = ∇_{θ^ℓ} L_t   for all layers ℓ

    for each layer ℓ do
        Compute gradient norm:
            n_t^ℓ = || g_t^ℓ ||_2

        Append n_t^ℓ to buffer B^ℓ

        if size(B^ℓ) == W then
            Compute gradient volatility:
                σ_t^ℓ = StdDev(B^ℓ)

            Compute layer pressure:
                p_t^ℓ = clip( 1 / (1 + k · σ_t^ℓ), p_min^ℓ, 1.0 )
        else
            p_t^ℓ = 1.0
        end if

        Apply layer pressure:
            g_t^ℓ ← p_t^ℓ · g_t^ℓ
    end for

    Update parameters:
        θ ← OPT(θ, {g_t^ℓ})
end for

Return trained parameters θ

---

### Computational Complexity

- Time per step:
    O(L) where L is number of layers
- Space:
    O(L · W) for per-layer buffers

SRLP-L enables fine-grained stability control with linear overhead in layer count.

