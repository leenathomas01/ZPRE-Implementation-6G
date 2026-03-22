![Version](https://img.shields.io/badge/version-v1.0-blue)

# ZPRE-Implementation  
**Adaptive Convergence Boundary for 6G ISAC**

A boundary study on the limits of adaptive interference cancellation under adversarial signal structures.

**Status: v1.0 - Stabilized Research Release**

---
> This repository preserves the complete experimental snapshot accompanying the research.
>
> The files represent the full implementation as it existed at the conclusion of the study.

---

## Core Insight

Adaptive interference cancellation assumes that interference is learnable.

This repository demonstrates a boundary condition:

There exists a class of signals whose structure prevents convergence of adaptive filters — even when the filters are nonlinear and theoretically capable of modeling the underlying dynamics.

The failure is not due to insufficient model capacity, but due to the rate and structure of discontinuities in the signal.

---

## Observed Boundary Behavior

- **Linear filters (FxLMS):**  
  Chaotic phase modulation (L1) collapses convergence (~3.5 dB SINR loss). The system cannot track discontinuities at τ ≈ 50 samples.

- **Nonlinear filters:**  
  - Volterra partially absorbs L1 (quadratic match) → reduced effect (~1.5 dB).  
  - KLMS fails to generalize; overfits local chaos.  
  - MLP absorbs L1 over time, but fails under L3/L4 (hidden state + orthogonal transitions).

- **Scaling behavior:**  
  Increasing model capacity (544 → ~49k parameters) yields marginal gains (~0.1–0.3 dB).

- **Discontinuity dominance:**  
  Increasing discontinuity rate (slow → extreme) produces larger effects (~1.6 dB) and prevents convergence.

### Boundary Condition

Adaptive systems fail when the rate of structural discontinuity exceeds their capacity to infer continuity.

This defines a practical limit of adaptive interference cancellation.

![Adaptive Convergence Boundary](adaptive_convergence_boundary.png)

---

## Repository Structure
```plaintext
ZPRE-Implementation/
├── README.md
├── requirements.txt
│
│  # Core system
├── FxLMS_UDP_Prototype.py       # FxLMS engine with UDP safety controls
├── ZPRE_Benchmarking.py         # Config sweeps, CSV logging, visualization
├── 6G_ISAC_Integration.py       # ISAC harness (KPIs, sensing, beamforming stubs)
│
│  # Adversarial anchor extension
├── ChaoticAnchor.py             # Multi-layer anchor generator (L1–L4)
├── NonlinearAdversary.py        # Volterra, KLMS, Online MLP filters
├── AnchorBenchmark.py           # Anchor vs FxLMS (linear adversary)
├── NonlinearBenchmark.py        # Anchor vs all adversary types
└── BoundaryProbe.py             # Scaled MLPs vs discontinuity rates
```

---

## Quick Start

```bash
# Baseline system
python FxLMS_UDP_Prototype.py        # FxLMS demo
python 6G_ISAC_Integration.py        # ISAC integration demo
python ZPRE_Benchmarking.py          # Config sweep + plots

# Adversarial anchor experiments
python AnchorBenchmark.py            # Anchor vs linear filter
python NonlinearBenchmark.py         # Anchor vs nonlinear adversaries
python BoundaryProbe.py              # Boundary mapping (scaled MLPs)
```

---

## Module Details

### Core System

| Module | Description |
|--------|-------------|
| `FxLMS_UDP_Prototype.py` | Filtered-x LMS with leakage, step clipping, NLMS normalization. Three modes (efficiency/balanced/enhance). Auto-escalation heuristic on residual variance. |
| `6G_ISAC_Integration.py` | ISAC-style harness: synthetic scene generation, canceller adapter, beamformer stub, matched-filter sensing, KPI computation (SINR, energy, latency, accuracy, coherence). |
| `ZPRE_Benchmarking.py` | Sweeps across modes and step sizes. CSV output + scatter plots. |

### Adversarial Anchor Extension

| Module | Description |
|--------|-------------|
| `ChaoticAnchor.py` | Four defense layers, independently toggleable: L1 (logistic map phase modulation), L2 (feedback-controlled chaos), L3 (private-key basis transitions), L4 (orthogonal policy jumps). |
| `NonlinearAdversary.py` | Three nonlinear adaptive filters: Volterra (2nd-order polynomial), Kernel LMS (RBF dictionary), Online MLP (backprop). Same step/process_block interface. |
| `AnchorBenchmark.py` | Anchor injected into FxLMS reference + error pathways. Measures SINR collapse, weight norm inflation, convergence curves. |
| `NonlinearBenchmark.py` | Cross-matrix: each anchor layer vs each adversary type. Produces the adversary comparison data. |
| `BoundaryProbe.py` | Variable-depth MLP (up to 3 layers, 128 hidden, 256-sample memory) vs anchor at four discontinuity rates. Maps the convergence boundary. |

---

## Extension Points

**Core system:**
- Replace `BeamformerStub` with THz/mmWave phase-array control
- Route canceller through photonic accelerator API
- Replace `SensingModule` with range-Doppler pipelines
- Add multi-channel / cross-node coherence

**Anchor extension:**
- LSTM/GRU adversary (recurrent memory across discontinuities)
- Orthogonal projection experiment (coupling vs modeling)
- Finer boundary sweep (interval 150–300, probability 0.03–0.07)

---

## Related Work

This repository applies bio-inspired optimization to 6G ISAC systems.

**For a complete catalog of related research:**  
[Research Index](https://github.com/leenathomas01/research-index)

**Thematically related:**
- [Zero Water AI Data Center](https://github.com/leenathomas01/zero-water-ai-dc) — Infrastructure optimization
- [Designing for Failure](https://github.com/leenathomas01/designing-for-failure-case-study) — Systems thinking on propagation medium trust

---
