"""
AnchorBenchmark.py — Adversarial anchor vs FxLMS stress test.
"""

from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

from FxLMS_UDP_Prototype import FxLMSConfig, FxLMSUDPEngine
from ChaoticAnchor import make_anchor


def generate_scene(N: int = 20000, seed: int = 42) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(N) / 16_000.0
    clean = 0.5 * np.sin(2 * np.pi * 200 * t)
    v = rng.normal(0, 1, size=N)
    b = np.array([1.0, -0.65, 0.3], dtype=np.float64)
    x = np.convolve(v, b, mode="same")
    p = np.array([0.8, 0.2, 0.05], dtype=np.float64)
    s = np.array([0.6, 0.3, 0.1], dtype=np.float64)
    d0 = clean + np.convolve(x, p, mode="same")
    return {"clean": clean, "x": x, "d0": d0, "s": s, "p": p, "t": t}


def run_attack(layer_name: str, signals: Dict[str, np.ndarray],
               alpha: float = 1.5, ref_cont: float = 0.6,
               mu: float = 0.02) -> Dict[str, Any]:
    N = signals["x"].size
    x0, d0, clean, s = signals["x"], signals["d0"], signals["clean"], signals["s"]

    cfg = FxLMSConfig(filter_len=128, base_mu=mu, leakage=5e-4, mode="balanced")
    eng = FxLMSUDPEngine(cfg, s_hat=s)

    is_baseline = (layer_name == "baseline")
    anchor = None if is_baseline else make_anchor(layer_name, alpha=alpha)

    e_out = np.zeros(N)
    weight_norms = []
    weight_snapshots = []

    for n in range(N):
        if is_baseline:
            a_n = 0.0
        else:
            prev_e = e_out[n - 1] if n > 0 else 0.0
            a_n = anchor.generate_sample(n, error=prev_e)

        x_in = x0[n] + ref_cont * a_n
        d_in = d0[n] + a_n
        _, _, e_n = eng.step(x_in, d_in)
        e_out[n] = e_n

        if n % 50 == 0:
            weight_norms.append({"t": int(n), "norm": float(np.linalg.norm(eng.w))})
        if n % (N // 6) == 0:
            weight_snapshots.append({"t": int(n), "w": eng.w[:32].tolist()})

    # Metrics
    sinr = FxLMSUDPEngine.estimate_sinr_gain_db(clean, pre=d0, post=e_out)
    epres = FxLMSUDPEngine.energy_preservation(d0 - clean, e_out - clean)

    W = 500
    conv = []
    for i in range(0, N - W, W // 4):
        seg = e_out[i:i + W]
        conv.append({"t": int(i + W // 2), "e_db": float(10 * np.log10(np.mean(seg**2) + 1e-12))})

    sinr_traj = []
    for i in range(W, N, W):
        sg = FxLMSUDPEngine.estimate_sinr_gain_db(clean[i-W:i], pre=d0[i-W:i], post=e_out[i-W:i])
        sinr_traj.append({"t": int(i), "sinr_db": float(sg)})

    # Weight velocity
    norms_arr = np.array([w["norm"] for w in weight_norms])
    vel = np.abs(np.diff(norms_arr))
    mean_vel = float(np.mean(vel)) if len(vel) > 0 else 0
    max_vel = float(np.max(vel)) if len(vel) > 0 else 0

    return {
        "layer": layer_name, "alpha": alpha, "ref_cont": ref_cont,
        "sinr_db": float(sinr), "energy_pres": float(epres),
        "final_mode": eng.cfg.mode, "w_norm_final": float(np.linalg.norm(eng.w)),
        "w_mean_velocity": mean_vel, "w_max_velocity": max_vel,
        "weight_norms": weight_norms, "weight_snapshots": weight_snapshots,
        "convergence": conv, "sinr_traj": sinr_traj,
        "anchor_diag": anchor.history if anchor else [],
    }


def run_full():
    signals = generate_scene(N=20000, seed=42)
    results = []

    # Baseline
    r = run_attack("baseline", signals, alpha=0)
    results.append(r)
    print(f"  baseline     SINR={r['sinr_db']:+.2f}  wnorm={r['w_norm_final']:.4f}  vel={r['w_mean_velocity']:.6f}")

    # Layer sweep
    for layer in ["none", "L1", "L1+L2", "L1+L2+L3", "full"]:
        r = run_attack(layer, signals, alpha=1.5)
        results.append(r)
        print(f"  {layer:<12} SINR={r['sinr_db']:+.2f}  wnorm={r['w_norm_final']:.4f}  vel={r['w_mean_velocity']:.6f}")

    # Alpha sweep on full
    for a in [0.3, 0.8, 1.5, 3.0, 5.0]:
        r = run_attack("full", signals, alpha=a)
        r["layer"] = f"full_a{a}"
        results.append(r)

    return results


if __name__ == "__main__":
    print("=== Anchor vs FxLMS ===")
    results = run_full()
    Path("anchor_results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {len(results)} results")
