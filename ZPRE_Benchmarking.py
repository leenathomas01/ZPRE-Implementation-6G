"""
ZPRE_Benchmarking.py
------------------------------------------------------------
Benchmarking harness for ZPRE-10 prototype modules.
Initial focus: FxLMS_UDP_Prototype.py adaptive interference
cancellation engine.

Purpose
-------
- Provide reproducible test environments (synthetic signals).
- Sweep across configs and modes to evaluate performance.
- Log results (CSV) with SINR/energy metrics.
- Plot quick visualizations for validation.

Framing: Pure R&D benchmark for wireless optimization /
adaptive algorithms (no speculative elements).

Author: The Lattice (Zee + collaborators)
License: Apache-2.0 (adjust per repo policy)
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any

# local import (assume in same repo)
from FxLMS_UDP_Prototype import FxLMSConfig, FxLMSUDPEngine


# ------------------------ signal generators ------------------------

def generate_test_signals(
    N: int = 20000, seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic test signals.
    clean : desired pilot/telemetry
    x     : reference interference
    d0    : error signal BEFORE cancellation
    s     : secondary path model
    """
    rng = np.random.default_rng(seed)
    t = np.arange(N) / 16_000.0

    # Clean sinusoidal pilot
    clean = 0.5 * np.sin(2 * np.pi * 200 * t)

    # Interference (colored noise)
    v = rng.normal(0, 1, size=N)
    b = np.array([1.0, -0.65, 0.3], dtype=np.float64)
    x = np.convolve(v, b, mode="same")

    # Primary path (unknown to controller)
    p = np.array([0.8, 0.2, 0.05], dtype=np.float64)

    # Secondary path (controller output to sensor)
    s = np.array([0.6, 0.3, 0.1], dtype=np.float64)

    d0 = clean + np.convolve(x, p, mode="same")

    return {"clean": clean, "x": x, "d0": d0, "s": s}


# ------------------------ benchmark runner ------------------------

def run_single_benchmark(cfg: FxLMSConfig, signals: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Run FxLMS engine on provided signals and return metrics."""
    eng = FxLMSUDPEngine(cfg, s_hat=signals["s"])
    _, _, e = eng.process_block(signals["x"], signals["d0"])
    d1 = e  # after cancellation

    sinr_gain = FxLMSUDPEngine.estimate_sinr_gain_db(
        signals["clean"], pre=signals["d0"], post=d1
    )
    energy_gain = FxLMSUDPEngine.energy_preservation(
        pre=signals["d0"] - signals["clean"], post=d1 - signals["clean"]
    )

    return {
        "mode": cfg.mode,
        "filter_len": cfg.filter_len,
        "base_mu": cfg.base_mu,
        "leakage": cfg.leakage,
        "sinr_gain_db": sinr_gain,
        "energy_preservation": energy_gain,
        "final_mode": eng.cfg.mode,  # may auto-escalate
    }


def sweep_configs(signals: Dict[str, np.ndarray]) -> list[Dict[str, Any]]:
    """Explore multiple configs for benchmarking."""
    results = []
    for mode in ["efficiency", "balanced", "enhance"]:
        for mu in [0.01, 0.02, 0.03]:
            cfg = FxLMSConfig(filter_len=128, base_mu=mu, leakage=5e-4, mode=mode)
            metrics = run_single_benchmark(cfg, signals)
            results.append(metrics)
    return results


# ------------------------ logging & plotting ------------------------

def write_csv(results: list[Dict[str, Any]], out_path: Path) -> None:
    keys = results[0].keys()
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


def plot_results(results: list[Dict[str, Any]]) -> None:
    """Quick scatter plot of SINR vs energy preservation."""
    modes = [r["mode"] for r in results]
    sinr = [r["sinr_gain_db"] for r in results]
    energy = [100.0 * r["energy_preservation"] for r in results]

    colors = {"efficiency": "blue", "balanced": "green", "enhance": "red"}
    plt.figure(figsize=(7, 5))
    for m, s, e in zip(modes, sinr, energy):
        plt.scatter(s, e, color=colors[m], label=m, alpha=0.7, s=60)
        plt.text(s, e, m[0].upper(), fontsize=9)

    plt.xlabel("SINR Gain (dB)")
    plt.ylabel("Energy Preservation (%)")
    plt.title("FxLMS UDP Benchmark Results")
    plt.grid(True, alpha=0.3)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, label=m)
               for m, c in colors.items()]
    plt.legend(handles=handles, title="Mode")
    plt.show()


# ------------------------ main harness ------------------------

def main() -> None:
    signals = generate_test_signals()
    results = sweep_configs(signals)

    out_path = Path("benchmark_results.csv")
    write_csv(results, out_path)
    print(f"[INFO] Wrote {len(results)} results to {out_path}")

    for r in results:
        print(
            f"Mode={r['mode']:<10} mu={r['base_mu']:.3f} | "
            f"SINR={r['sinr_gain_db']:+6.2f} dB | "
            f"EnergyPres={100.0*r['energy_preservation']:5.1f}% | "
            f"FinalMode={r['final_mode']}"
        )

    plot_results(results)


if __name__ == "__main__":
    main()
