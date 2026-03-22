"""
ChaoticAnchor.py
------------------------------------------------------------
Multi-layer adversarial anchor system for disrupting adaptive
interference cancellation (FxLMS / nonlinear filters).

Implements four defense layers discussed in the ZPRE-10 framework:

  Layer 1 — Structural:   Chaotic phase modulation (logistic map)
  Layer 2 — Dynamic:      Feedback-controlled chaos (adaptive tau/theta)
  Layer 3 — Hidden State:  Exogenous semantic driver (private key transitions)
  Layer 4 — Epistemic:    Orthogonal anchor policy (TRNG-style switching)

Each layer can be enabled independently for benchmarking.

Author: Zee + collaborators (The Lattice)
License: Apache-2.0
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


# -------------------- Configuration --------------------

@dataclass
class AnchorConfig:
    """Configuration for the chaotic anchor system."""
    # Carrier (Layer 0: local coherence)
    fc: float = 200.0              # carrier frequency Hz
    fs: float = 16_000.0           # sample rate

    # Layer 1: Structural chaos (logistic map phase modulation)
    enable_chaos: bool = True
    r: float = 3.99                # logistic map parameter (deep chaos)
    tau_samples: int = 50          # samples between chaotic phase jumps
    phase_scale: float = np.pi     # max phase shift severity (radians)

    # Layer 2: Adaptive feedback
    enable_adaptive: bool = False
    lambda_smooth: float = 0.99    # error energy smoothing factor
    e_target: float = 0.1          # survival threshold for error energy
    tau_min: int = 5               # minimum tau (maximum chaos rate)
    tau_max: int = 200             # maximum tau (minimum chaos rate)
    tau_adapt_rate: float = 0.3    # how fast tau adjusts
    theta_adapt_rate: float = 0.05 # how fast phase_scale adjusts

    # Layer 3: Private key transitions (keyed PRNG)
    enable_private_key: bool = False
    key_seed: int = 31415          # private key (unknown to observer)
    key_transition_interval: int = 500  # samples between key-driven jumps
    num_basis_states: int = 5      # number of orthogonal anchor bases

    # Layer 4: Orthogonal policy (full TRNG-style)
    enable_orthogonal: bool = False
    orthogonal_switch_prob: float = 0.02  # per-sample probability of state jump

    # Injection strength
    alpha: float = 1.0             # anchor injection amplitude


class ChaoticAnchor:
    """
    Multi-layer adversarial anchor generator.
    
    Produces a signal that is locally coherent (you can hold it)
    but globally unlearnable by adaptive systems.
    """

    def __init__(self, cfg: AnchorConfig):
        self.cfg = cfg
        self.sample_count = 0

        # Layer 1: logistic map state
        self._chaos_x = 0.5 + 0.01 * np.random.rand()  # slight jitter off fixed point
        self._current_phase = 0.0

        # Layer 2: adaptive state
        self._error_energy_avg = 1.0
        self._adaptive_tau = float(cfg.tau_samples)
        self._adaptive_theta = cfg.phase_scale

        # Layer 3: private key PRNG (separate from observable randomness)
        self._key_rng = np.random.default_rng(cfg.key_seed)
        self._basis_phases = self._key_rng.uniform(0, 2 * np.pi, size=cfg.num_basis_states)
        self._basis_freqs = cfg.fc * (1.0 + 0.3 * self._key_rng.uniform(-1, 1, size=cfg.num_basis_states))
        self._current_basis = 0
        self._key_offset = 0.0

        # Layer 4: orthogonal library (distinct waveform generators)
        self._ortho_rng = np.random.default_rng(cfg.key_seed + 777)
        self._ortho_state = 0

        # Diagnostics
        self.history: List[dict] = []

    def reset(self):
        self.__init__(self.cfg)

    # -------------------- Core generation --------------------

    def generate_sample(self, t_index: int, error: float = 0.0) -> float:
        """
        Generate one anchor sample at time index t_index.
        
        Args:
            t_index: sample index
            error: current system residual error (for Layer 2 feedback)
        
        Returns:
            anchor signal value
        """
        self.sample_count += 1
        t = t_index / self.cfg.fs
        phase = 0.0

        # --- Layer 1: Structural chaos ---
        if self.cfg.enable_chaos:
            tau = int(self._adaptive_tau) if self.cfg.enable_adaptive else self.cfg.tau_samples
            tau = max(1, tau)
            if self.sample_count % tau == 0:
                self._chaos_x = self.cfg.r * self._chaos_x * (1.0 - self._chaos_x)
            theta = self._adaptive_theta if self.cfg.enable_adaptive else self.cfg.phase_scale
            phase += theta * self._chaos_x

        # --- Layer 2: Adaptive feedback ---
        if self.cfg.enable_adaptive:
            self._update_adaptive(error)

        # --- Layer 3: Private key transitions ---
        fc = self.cfg.fc
        if self.cfg.enable_private_key:
            if self.sample_count % self.cfg.key_transition_interval == 0:
                self._current_basis = int(self._key_rng.integers(0, self.cfg.num_basis_states))
                self._key_offset = self._key_rng.uniform(0, 2 * np.pi)
            fc = self._basis_freqs[self._current_basis]
            phase += self._basis_phases[self._current_basis] + self._key_offset

        # --- Layer 4: Orthogonal policy ---
        if self.cfg.enable_orthogonal:
            if self._ortho_rng.random() < self.cfg.orthogonal_switch_prob:
                self._ortho_state = int(self._ortho_rng.integers(0, self.cfg.num_basis_states))
                # Discontinuous phase/freq jump
                fc = self._basis_freqs[self._ortho_state]
                phase += self._ortho_rng.uniform(0, 2 * np.pi)

        self._current_phase = phase
        signal = self.cfg.alpha * np.sin(2.0 * np.pi * fc * t + phase)

        # Log diagnostics (sparse)
        if self.sample_count % 100 == 0:
            self.history.append({
                "t": t_index,
                "chaos_x": self._chaos_x,
                "tau_eff": self._adaptive_tau if self.cfg.enable_adaptive else self.cfg.tau_samples,
                "theta_eff": self._adaptive_theta if self.cfg.enable_adaptive else self.cfg.phase_scale,
                "basis": self._current_basis if self.cfg.enable_private_key else -1,
                "error_energy": self._error_energy_avg,
            })

        return signal

    def generate_block(self, N: int, start_index: int = 0, errors: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate a block of anchor samples."""
        out = np.zeros(N, dtype=np.float64)
        for i in range(N):
            e = errors[i] if errors is not None else 0.0
            out[i] = self.generate_sample(start_index + i, error=e)
        return out

    # -------------------- Layer 2 internals --------------------

    def _update_adaptive(self, error: float):
        """Feedback-controlled chaos: adjust tau and theta based on error energy."""
        lam = self.cfg.lambda_smooth
        self._error_energy_avg = lam * self._error_energy_avg + (1.0 - lam) * error ** 2

        if self._error_energy_avg < self.cfg.e_target:
            # System is locking on — increase chaos
            self._adaptive_tau = max(
                self.cfg.tau_min,
                self._adaptive_tau - self.cfg.tau_adapt_rate
            )
            self._adaptive_theta = min(
                2.0 * self.cfg.phase_scale,  # cap at 2x base
                self._adaptive_theta + self.cfg.theta_adapt_rate
            )
        else:
            # System is disrupted — stabilize for coherence
            self._adaptive_tau = min(
                self.cfg.tau_max,
                self._adaptive_tau + self.cfg.tau_adapt_rate * 0.5
            )
            self._adaptive_theta = max(
                0.3 * self.cfg.phase_scale,  # floor at 30% base
                self._adaptive_theta - self.cfg.theta_adapt_rate * 0.5
            )


# -------------------- Anchor presets for benchmarking --------------------

def make_anchor(layer: str, alpha: float = 0.8, **overrides) -> ChaoticAnchor:
    """
    Factory for common anchor configurations.
    
    layer options:
        'none'       — pure carrier, no defense
        'L1'         — structural chaos only
        'L1+L2'      — chaos + adaptive feedback
        'L1+L2+L3'   — chaos + adaptive + private key
        'full'       — all four layers
    """
    base = AnchorConfig(alpha=alpha)

    if layer == "none":
        base.enable_chaos = False
        base.enable_adaptive = False
        base.enable_private_key = False
        base.enable_orthogonal = False
    elif layer == "L1":
        base.enable_chaos = True
    elif layer == "L1+L2":
        base.enable_chaos = True
        base.enable_adaptive = True
    elif layer == "L1+L2+L3":
        base.enable_chaos = True
        base.enable_adaptive = True
        base.enable_private_key = True
    elif layer == "full":
        base.enable_chaos = True
        base.enable_adaptive = True
        base.enable_private_key = True
        base.enable_orthogonal = True
    else:
        raise ValueError(f"Unknown layer preset: {layer}")

    for k, v in overrides.items():
        if hasattr(base, k):
            setattr(base, k, v)

    return ChaoticAnchor(base)
