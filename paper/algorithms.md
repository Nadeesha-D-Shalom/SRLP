## Algorithm 1: SRLP (Global)

Input: model parameters θ, optimizer Opt, window size W, sensitivity k  
Initialize loss buffer B ← empty

For each training step t:
    Compute loss L_t
    Append L_t to B
    If |B| = W:
        σ_t ← StdDev(B)
        p_t ← clip(1 / (1 + k·σ_t), p_min, 1.0)
    Else:
        p_t ← 1.0
    Compute gradients ∇θ_t
    Scale gradients: ∇θ_t ← p_t · ∇θ_t
    Update θ using Opt

---

## Algorithm 2: SRLP-L (Layer-wise)

For each layer ℓ:
    Maintain gradient norm buffer B_ℓ

For each training step t:
    For each layer ℓ:
        Compute gradient norm g_t^ℓ
        Append g_t^ℓ to B_ℓ
        If |B_ℓ| = W:
            σ_t^ℓ ← StdDev(B_ℓ)
            p_t^ℓ ← clip(1 / (1 + k·σ_t^ℓ), p_min^ℓ, 1.0)
        Else:
            p_t^ℓ ← 1.0
        Scale layer gradients: ∇θ_t^ℓ ← p_t^ℓ · ∇θ_t^ℓ
