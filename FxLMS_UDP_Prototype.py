"""
FxLMS_UDP_Prototype.py
------------------------------------------------------------
Adaptive interference cancellation prototype for wireless /
acoustic-style feedforward ANC using the Filtered-x LMS (FxLMS)
algorithm. Framed for 6G ISAC-style experimentation.

This module is **research scaffolding**:
- Safe, front-facing (no speculative content).
- Minimal deps (NumPy only).
- Clear extension points for future modules:
  * Antibody_Memory_System (bio-inspired memory)
  * TDA_VoidFilling       (cascade topology detection)
  * RL_PolicyEngine       (mode policy optimization)

Key ideas implemented here
- FxLMS core loop with secondary-path compensation.
- "Unified Dampening Protocol (UDP)" safety knobs:
    • leakage (weight decay)
    • normalized step size (power-based)
    • step clipping (stability guard)
- Three operational modes (efficiency/balanced/enhance).
- Simple cascade-risk heuristic for auto-escalation.
- Lightweight benchmarking helper in __main__.

Author: The Lattice (Zee + collaborators)
License: Apache-2.0 (adjust per repo policy)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np


# --------- Modes (public, R&D-facing; no consciousness terminology) ---------

MODE_PARAMS: Dict[str, Dict[str, float]] = {
    # Conservative energy use, slower convergence
    "efficiency": {"mu_scale": 0.75, "sinr_weight": 1.0},
    # Default operating point
    "balanced":   {"mu_scale": 1.00, "sinr_weight": 1.5},
    # Fastest convergence, more aggressive adaptation
    "enhance":    {"mu_scale": 1.25, "sinr_weight": 2.0},
}


@dataclass
class FxLMSConfig:
    filter_len: int = 128          # FIR length
    base_mu: float = 0.01          # base step size (scaled by mode)
    leakage: float = 0.0005        # weight decay per step (UDP safety)
    step_clip: float = 0.05        # absolute clip on effective mu*e
    normalize: bool = True         # NLMS-style normalization
    mode: str = "balanced"         # 'efficiency' | 'balanced' | 'enhance'
    # auto-escalation thresholds (heuristic; tune per scenario)
    cascade_prob_threshold: float = 0.10
    residual_std_window: int = 256
    escalate_std_ratio: float = 1.35  # if residual std rises vs. baseline by this factor → escalate
    # resources knob for potential downshift (not enforced here; placeholder)
    resource_level: float = 1.0    # 0..1 (future: used to auto-drop to 'efficiency')


class FxLMSUDPEngine:
    """
    FxLMS engine with UDP-style safety controls and simple mode logic.

    Signal model (standard feedforward ANC/wireless IC):
      x[n] : reference interference (correlated with disturbance)
      d[n] : error measurement (desired + disturbance at sensor)
      y[n] : controller output (to cancel disturbance via secondary path)
      e[n] : residual error = d[n] - y_through_secondary[n]

    Secondary-path model s_hat is assumed known/identified for Fx filtering.
    """

    def __init__(self, config: FxLMSConfig, s_hat: np.ndarray):
        assert config.mode in MODE_PARAMS, f"Unknown mode '{config.mode}'"
        self.cfg = config
        self.s_hat = np.asarray(s_hat, dtype=np.float64).copy()
        assert self.s_hat.ndim == 1 and self.s_hat.size >= 1

        self.L = self.cfg.filter_len
        self.w = np.zeros(self.L, dtype=np.float64)

        # ring buffers
        self._xbuf = np.zeros(self.L, dtype=np.float64)             # raw reference buffer
        self._xfbuf = np.zeros(self.L + self.s_hat.size - 1)        # filtered-x (convolution) temp
        self._ybuf = np.zeros(self.s_hat.size, dtype=np.float64)    # controller output for sec. path

        # residual stats for simple cascade heuristic
        self._res_hist = []

        # mode derived parameters
        self._mu_scale = MODE_PARAMS[self.cfg.mode]["mu_scale"]

    # --------------------------- public API ---------------------------------

    def reset(self) -> None:
        self.w.fill(0.0)
        self._xbuf.fill(0.0)
        self._xfbuf.fill(0.0)
        self._ybuf.fill(0.0)
        self._res_hist.clear()

    def set_mode(self, mode: str) -> None:
        """Switch operating mode."""
        assert mode in MODE_PARAMS, f"Unknown mode '{mode}'"
        self.cfg.mode = mode
        self._mu_scale = MODE_PARAMS[mode]["mu_scale"]

    def step(self, x: float, d: float) -> Tuple[float, float, float]:
        """
        Process one sample.

        Args
        ----
        x : reference interference sample
        d : error sensor sample (desired + interference)

        Returns
        -------
        y       : controller output (pre-secondary-path)
        y_thru  : controller output after secondary path (contributes to cancellation)
        e       : residual error
        """
        # update x buffer (most recent at index 0)
        self._xbuf[1:] = self._xbuf[:-1]
        self._xbuf[0] = x

        # controller output y = w^T * x_vec
        y = float(np.dot(self.w, self._xbuf))

        # push y through secondary path (convolution with s_hat, streaming)
        self._ybuf[1:] = self._ybuf[:-1]
        self._ybuf[0] = y
        y_thru = float(np.dot(self.s_hat, self._ybuf))

        # residual error
        e = d - y_thru

        # Filtered-x: x filtered by s_hat (approx s)
        # Efficient streaming conv for the needed L taps:
        # compute convolution of current x-buffer with s_hat and take first L samples
        # (We maintain a temp conv buffer; for clarity we recompute.)
        xf_full = np.convolve(self._xbuf, self.s_hat, mode="full")
        xf_vec = xf_full[: self.L]  # aligned with w and xbuf

        # Normalization (prevent gradient explosion on large xf power)
        denom = np.dot(xf_vec, xf_vec) + 1e-9 if self.cfg.normalize else 1.0

        # Effective step
        mu_eff = (self.cfg.base_mu * self._mu_scale) / denom

        # UDP step clipping (acts like a safety limiter)
        grad = e * xf_vec
        grad_norm = np.clip(grad, -self.cfg.step_clip, self.cfg.step_clip)

        # weight update with leakage
        self.w = (1.0 - self.cfg.leakage) * self.w + 2.0 * mu_eff * grad_norm

        # --- simple cascade-risk tracking (heuristic) -----------------------
        self._track_residual_stats(e)

        return y, y_thru, e

    def process_block(self, x: np.ndarray, d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized convenience wrapper around step().
        """
        x = np.asarray(x, dtype=np.float64)
        d = np.asarray(d, dtype=np.float64)
        assert x.shape == d.shape
        N = x.size

        y = np.zeros(N, dtype=np.float64)
        y_thru = np.zeros(N, dtype=np.float64)
        e = np.zeros(N, dtype=np.float64)

        for n in range(N):
            y[n], y_thru[n], e[n] = self.step(x[n], d[n])
            # Auto-escalate mode if residual behavior suggests a forming cascade
            if self._should_escalate():
                self.set_mode("enhance")

        return y, y_thru, e

    # --------------------------- helpers ------------------------------------

    def _track_residual_stats(self, e: float) -> None:
        self._res_hist.append(e)
        if len(self._res_hist) > self.cfg.residual_std_window * 2:
            self._res_hist = self._res_hist[-self.cfg.residual_std_window * 2 :]

    def _should_escalate(self) -> bool:
        """
        Heuristic: if recent residual variance grows vs. baseline, treat as
        elevated cascade probability and request 'enhance' mode.

        NOTE: This is intentionally simple. A future TDA_VoidFilling module
        should replace this with topological triggers on system state.
        """
        W = self.cfg.residual_std_window
        if len(self._res_hist) < 2 * W:
            return False
        recent = np.array(self._res_hist[-W:], dtype=np.float64)
        prior = np.array(self._res_hist[-2 * W : -W], dtype=np.float64)
        prior_std = np.std(prior) + 1e-9
        ratio = float(np.std(recent) / prior_std)
        return ratio >= self.cfg.escalate_std_ratio

    # ---------------------- benchmarking utilities --------------------------

    @staticmethod
    def estimate_sinr_gain_db(clean: np.ndarray, pre: np.ndarray, post: np.ndarray) -> float:
        """
        Approximate SINR improvement using known clean signal for simulation:
        pre  = clean + noise_before
        post = clean + noise_after
        """
        clean = np.asarray(clean, dtype=np.float64)
        pre = np.asarray(pre, dtype=np.float64)
        post = np.asarray(post, dtype=np.float64)

        s_pow = np.mean(clean**2) + 1e-12
        n_pre = pre - clean
        n_post = post - clean
        sinr_pre = s_pow / (np.mean(n_pre**2) + 1e-12)
        sinr_post = s_pow / (np.mean(n_post**2) + 1e-12)
        return 10.0 * np.log10(sinr_post / (sinr_pre + 1e-18))

    @staticmethod
    def energy_preservation(pre: np.ndarray, post: np.ndarray) -> float:
        """
        Relative reduction in residual energy (1 - E[post^2]/E[pre^2]).
        """
        pre_e = np.mean(np.asarray(pre, dtype=np.float64) ** 2) + 1e-12
        post_e = np.mean(np.asarray(post, dtype=np.float64) ** 2) + 1e-12
        return max(0.0, 1.0 - post_e / pre_e)


# ------------------------------- demo ---------------------------------------

def _demo(seed: int = 7) -> None:
    """
    Minimal self-test:
    - generate clean signal + correlated interference
    - apply FxLMS cancellation
    - print crude metrics for quick sanity checking
    """
    rng = np.random.default_rng(seed)
    N = 20000
    t = np.arange(N) / 16_000.0

    # Clean "desired" component (e.g., pilot/telemetry)
    clean = 0.5 * np.sin(2 * np.pi * 200 * t)

    # Interference at reference (x): colored noise
    v = rng.normal(0, 1, size=N)
    b = np.array([1.0, -0.65, 0.3], dtype=np.float64)  # simple AR-ish color
    x = np.convolve(v, b, mode="same")

    # Primary path from x to disturbance at error sensor (unknown to controller)
    p = np.array([0.8, 0.2, 0.05], dtype=np.float64)

    # Secondary path s (from controller output to error sensor)
    s = np.array([0.6, 0.3, 0.1], dtype=np.float64)

    # Error sensor BEFORE cancellation: d0 = clean + (x * p)
    d0 = clean + np.convolve(x, p, mode="same")

    # Engine with s_hat ~= s (identified secondary path model)
    cfg = FxLMSConfig(filter_len=128, base_mu=0.03, leakage=5e-4, mode="balanced")
    eng = FxLMSUDPEngine(cfg, s_hat=s)

    # Run adaptation (controller observes x, d)
    _, _, e = eng.process_block(x, d0)

    # AFTER cancellation, measurement is residual error e (≈ clean + reduced noise)
    d1 = e

    sinr_gain = FxLMSUDPEngine.estimate_sinr_gain_db(clean, pre=d0, post=d1)
    energy_gain = FxLMSUDPEngine.energy_preservation(pre=d0 - clean, post=d1 - clean)

    print("=== FxLMS UDP Prototype (demo) ===")
    print(f"Mode: {cfg.mode}")
    print(f"Estimated SINR gain: {sinr_gain:6.2f} dB")
    print(f"Residual energy preservation: {100.0 * energy_gain:5.1f}%")
    print(f"Escalated mode? -> {eng.cfg.mode!r} (auto-escalation may switch to 'enhance')")


if __name__ == "__main__":
    _demo()
