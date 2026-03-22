"""
6G_ISAC_Integration.py
------------------------------------------------------------
Standards-facing harness for integrating ZPRE-10 prototype modules
(e.g., FxLMS_UDP_Prototype.py) into an ISAC-style workflow.

Scope (purely technical):
- Define ISAC-like configs and KPIs (SINR, sensing accuracy, latency).
- Provide adapters to plug in cancellation / beamforming / sensing logic.
- Run compliance-style checks against target thresholds inspired by
  emerging 6G discussions (ETSI/3GPP/ITU) WITHOUT implying certification.

Design goals:
- Minimal deps (numpy only).
- Clear extension points for hardware (photonic accel, THz arrays).
- Runnable demo producing KPI summary.

Author: The Lattice (Zee + collaborators)
License: Apache-2.0 (adjust per repo policy)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import time
import numpy as np

# Local prototype (must be in same repo)
from FxLMS_UDP_Prototype import FxLMSConfig, FxLMSUDPEngine


# -------------------------- ISAC-like config --------------------------

@dataclass
class ISACTargets:
    """Target KPIs (tunable per project/partner)."""
    min_sinr_gain_db: float = 20.0
    min_energy_preservation: float = 0.80      # 80%
    max_end_to_end_latency_ms: float = 1.0     # < 1 ms budget for control loop
    min_sensing_accuracy: float = 0.90         # 90% detection/ID accuracy
    min_multi_node_coherence: float = 0.92     # 92% SINR coherence (proxy)

@dataclass
class ISACConfig:
    """System-level configuration."""
    fs_hz: float = 16_000.0
    block_len: int = 2048
    carrier_hz: float = 200.0
    noise_std: float = 1.0
    rng_seed: int = 7

    # Sensing scene
    num_targets: int = 2
    target_delays: Tuple[int, ...] = (40, 95)  # samples
    target_gains: Tuple[float, ...] = (0.9, 0.6)

    # Integration knobs
    canceller_mode: str = "balanced"   # 'efficiency'|'balanced'|'enhance'
    canceller_mu: float = 0.02
    canceller_len: int = 128
    leakage: float = 5e-4

    # Secondary path estimate (controller output→sensor)
    sec_path: Tuple[float, ...] = (0.6, 0.3, 0.1)

    # “Hardware” placeholders (interfaces only)
    use_photonic_accel: bool = False
    use_thz_phase_array: bool = False


# -------------------------- Synthetic ISAC scene --------------------------

def synth_scene(cfg: ISACConfig) -> Dict[str, np.ndarray]:
    """
    Create a simple joint comm-sense scene:
    - clean pilot/telemetry signal
    - interference (colored noise)
    - sensing echo from sparse targets (convolutional delays)
    - pre-cancellation measurement (clean + interference + echoes)
    """
    rng = np.random.default_rng(cfg.rng_seed)
    N = cfg.block_len
    t = np.arange(N) / cfg.fs_hz

    clean = 0.5 * np.sin(2 * np.pi * cfg.carrier_hz * t)

    # Interference (colored)
    v = rng.normal(0, cfg.noise_std, size=N)
    b = np.array([1.0, -0.65, 0.3], dtype=np.float64)
    x = np.convolve(v, b, mode="same")

    # Primary coupling path (unknown)
    p = np.array([0.8, 0.2, 0.05], dtype=np.float64)

    # Sensing echoes (sparse taps)
    echo = np.zeros_like(clean)
    for d, g in zip(cfg.target_delays, cfg.target_gains):
      if 0 <= d < N:
        echo[d] += g
    echo = np.convolve(clean, echo, mode="same")

    d0 = clean + np.convolve(x, p, mode="same") + echo
    s_hat = np.array(cfg.sec_path, dtype=np.float64)
    return {"clean": clean, "x": x, "d0": d0, "echo": echo, "s_hat": s_hat}


# -------------------------- Adapters / Modules --------------------------

class CancellerAdapter:
    """Wraps the FxLMS canceller for ISAC harness."""
    def __init__(self, cfg: ISACConfig):
        self.cfg = cfg
        fx = FxLMSConfig(
            filter_len=cfg.canceller_len,
            base_mu=cfg.canceller_mu,
            leakage=cfg.leakage,
            mode=cfg.canceller_mode,
        )
        self.engine = FxLMSUDPEngine(fx, s_hat=np.array(cfg.sec_path, dtype=np.float64))

    def run(self, x: np.ndarray, d0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start = time.perf_counter()
        y, e, _ = self.engine.process_block(x, d0)
        latency_ms = (time.perf_counter() - start) * 1e3
        return e, y  # e: residual (post), y: control, (latency handled by KPI stage)


class BeamformerStub:
    """
    Placeholder for THz/mmWave beam steering control.
    For now, returns identity/neutral behavior and a coherence proxy.
    """
    def __init__(self, use_thz: bool):
        self.use_thz = use_thz

    def apply(self, signal: np.ndarray) -> Tuple[np.ndarray, float]:
        # In the real system, this would apply phase-array steering.
        # We return signal unchanged and a synthetic “coherence” proxy:
        coherence = 0.94 if self.use_thz else 0.90
        return signal, coherence


class SensingModule:
    """
    Simple matched-filter detector against the known pilot to estimate
    sensing 'accuracy' (did we correctly localize echoes/delays?).
    """
    def __init__(self, pilot: np.ndarray, delays: Tuple[int, ...], tol: int = 3):
        self.pilot = pilot
        self.expected = set(delays)
        self.tol = tol

    def detect_delays(self, measured: np.ndarray, k: int = 2) -> set[int]:
        # naive detection: cross-correlate with pilot, pick top-k peaks
        c = np.correlate(measured, self.pilot, mode="full")
        mid = (len(c) - 1) // 2  # zero lag index
        # restrict search to plausible positive lags
        segment = c[mid:mid + 256]
        peaks = np.argpartition(segment, -k)[-k:]
        return set(int(p) for p in np.sort(peaks))

    def accuracy(self, measured: np.ndarray) -> float:
        found = self.detect_delays(measured, k=len(self.expected))
        score = 0
        for e in self.expected:
            if any(abs(e - f) <= self.tol for f in found):
                score += 1
        return score / max(1, len(self.expected))


# -------------------------- KPI computation --------------------------

@dataclass
class KPIResult:
    sinr_gain_db: float
    energy_preservation: float
    latency_ms: float
    sensing_accuracy: float
    multi_node_coherence: float
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)


def compute_kpis(
    clean: np.ndarray,
    pre: np.ndarray,
    post: np.ndarray,
    control_latency_ms: float,
    sensing_accuracy: float,
    coherence_proxy: float,
    targets: ISACTargets,
) -> KPIResult:
    sinr_gain = FxLMSUDPEngine.estimate_sinr_gain_db(clean, pre=pre, post=post)
    energy_gain = FxLMSUDPEngine.energy_preservation(pre - clean, post - clean)

    passed = (
        (sinr_gain >= targets.min_sinr_gain_db) and
        (energy_gain >= targets.min_energy_preservation) and
        (control_latency_ms <= targets.max_end_to_end_latency_ms) and
        (sensing_accuracy >= targets.min_sensing_accuracy) and
        (coherence_proxy >= targets.min_multi_node_coherence)
    )

    return KPIResult(
        sinr_gain_db=sinr_gain,
        energy_preservation=energy_gain,
        latency_ms=control_latency_ms,
        sensing_accuracy=sensing_accuracy,
        multi_node_coherence=coherence_proxy,
        passed=passed,
        details={
            "thresholds": targets,
        },
    )


# -------------------------- Integration flow --------------------------

def run_isac_integration(cfg: ISACConfig, targets: Optional[ISACTargets] = None) -> KPIResult:
    targets = targets or ISACTargets()
    scene = synth_scene(cfg)

    # Canceller path
    canceller = CancellerAdapter(cfg)
    t0 = time.perf_counter()
    post, control = canceller.run(scene["x"], scene["d0"])
    control_latency_ms = (time.perf_counter() - t0) * 1e3

    # Beamforming / coherence proxy
    beam = BeamformerStub(cfg.use_thz_phase_array)
    post_bf, coherence = beam.apply(post)

    # Sensing accuracy (after cancellation → cleaner echoes)
    sense = SensingModule(scene["clean"], delays=cfg.target_delays)
    accuracy = sense.accuracy(post_bf)

    # KPIs
    kpi = compute_kpis(
        clean=scene["clean"],
        pre=scene["d0"],
        post=post_bf,
        control_latency_ms=control_latency_ms,
        sensing_accuracy=accuracy,
        coherence_proxy=coherence,
        targets=targets,
    )

    # Attach debug traces (trimmed) for downstream benchmarking
    kpi.details.update({
        "mode": canceller.engine.cfg.mode,
        "latency_ms_measured": control_latency_ms,
        "post_head": post_bf[:8].tolist(),
        "control_head": control[:8].tolist(),
    })
    return kpi


# -------------------------- CLI / demo --------------------------

def _pretty_pct(x: float) -> str:
    return f"{100.0*x:5.1f}%"

def main() -> None:
    cfg = ISACConfig(
        canceller_mode="balanced",
        use_thz_phase_array=True,      # toggle to see coherence effect
        use_photonic_accel=False,      # stubbed; see TODO below
    )
    targets = ISACTargets(
        min_sinr_gain_db=20.0,
        min_energy_preservation=0.80,
        max_end_to_end_latency_ms=1.0,
        min_sensing_accuracy=0.90,
        min_multi_node_coherence=0.92,
    )

    kpi = run_isac_integration(cfg, targets)

    print("=== ISAC Integration KPI Summary ===")
    print(f" Mode:                 {kpi.details['mode']}")
    print(f" SINR Gain:            {kpi.sinr_gain_db:+6.2f} dB")
    print(f" Energy Preservation:  {_pretty_pct(kpi.energy_preservation)}")
    print(f" Control Latency:      {kpi.latency_ms:5.2f} ms")
    print(f" Sensing Accuracy:     {_pretty_pct(kpi.sensing_accuracy)}")
    print(f" Multi-node Coherence: {_pretty_pct(kpi.multi_node_coherence)}")
    print(f" PASSED:               {kpi.passed}")
    print("------------------------------------")
    print(" Notes:")
    print("  - BeamformerStub and photonic acceleration are placeholders.")
    print("  - Tune ISACTargets to partner requirements.")
    print("  - For full studies, use ZPRE_Benchmarking.py to sweep configs.")

    # TODO hooks (for contributors):
    # 1) Replace BeamformerStub with real THz/mmWave phase-array control.
    # 2) Route canceller.run() through a photonic accelerator API to
    #    measure sub-millisecond loop execution.
    # 3) Replace SensingModule with ISAC-compliant range-Doppler pipelines.
    # 4) Add multi-channel scenarios and coherence measurements across nodes.


if __name__ == "__main__":
    main()
