"""
NonlinearBenchmark.py
------------------------------------------------------------
The real test: Can the Chaotic Anchor survive against nonlinear
adaptive systems that can theoretically model the logistic map?

Tests each anchor layer against:
  - FxLMS (linear baseline)
  - Volterra (2nd order polynomial — can learn logistic map directly)
  - KLMS (kernel method — universal approximator)
  - Online MLP (neural net — universal approximator)

Key question: Does Layer 3/4 (private key / orthogonal policy)
actually provide defense that L1 cannot, once the adversary is
nonlinear?
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

from FxLMS_UDP_Prototype import FxLMSConfig, FxLMSUDPEngine
from ChaoticAnchor import make_anchor
from NonlinearAdversary import make_adversary


def generate_scene(N: int = 12000, seed: int = 42):
    rng = np.random.default_rng(seed)
    t = np.arange(N) / 16_000.0
    clean = 0.5 * np.sin(2 * np.pi * 200 * t)
    v = rng.normal(0, 1, size=N)
    b = np.array([1.0, -0.65, 0.3])
    x = np.convolve(v, b, mode="same")
    p = np.array([0.8, 0.2, 0.05])
    s = np.array([0.6, 0.3, 0.1])
    d0 = clean + np.convolve(x, p, mode="same")
    return {"clean": clean, "x": x, "d0": d0, "s": s, "t": t}


def run_matchup(adversary_kind: str, anchor_layer: str,
                signals: Dict[str, np.ndarray],
                alpha: float = 1.5, ref_cont: float = 0.6) -> Dict[str, Any]:
    N = signals["x"].size
    x0, d0, clean = signals["x"], signals["d0"], signals["clean"]

    # Build adversary
    adv = make_adversary(adversary_kind, s_hat=signals["s"])
    is_fxlms = adversary_kind == "fxlms"

    # Build anchor
    is_baseline = anchor_layer == "baseline"
    anchor = None if is_baseline else make_anchor(anchor_layer, alpha=alpha)

    e_out = np.zeros(N)
    w_norms = []

    for n in range(N):
        if is_baseline:
            a_n = 0.0
        else:
            prev_e = e_out[n - 1] if n > 0 else 0.0
            a_n = anchor.generate_sample(n, error=prev_e)

        x_in = x0[n] + ref_cont * a_n
        d_in = d0[n] + a_n

        if is_fxlms:
            _, _, e_n = adv.step(x_in, d_in)
        else:
            _, _, e_n = adv.step(x_in, d_in)

        e_out[n] = e_n

        if n % 100 == 0:
            wn = float(np.linalg.norm(adv.w))
            w_norms.append({"t": int(n), "n": round(wn, 5)})

    sinr = FxLMSUDPEngine.estimate_sinr_gain_db(clean, pre=d0, post=e_out)

    # Convergence curve (windowed)
    W = 500
    conv = []
    for i in range(0, N - W, W // 2):
        seg = e_out[i:i + W]
        conv.append({"t": int(i + W // 2),
                      "e": round(float(10 * np.log10(np.mean(seg**2) + 1e-12)), 2)})

    return {
        "adversary": adversary_kind,
        "anchor": anchor_layer,
        "alpha": alpha,
        "sinr_db": round(float(sinr), 2),
        "w_norm_final": round(float(np.linalg.norm(adv.w)), 4),
        "weight_norms": w_norms,
        "convergence": conv,
        "N": N,
    }


def run_full():
    signals = generate_scene(N=12000, seed=42)
    adversaries = ["fxlms", "volterra", "klms", "mlp"]
    anchor_layers = ["baseline", "L1", "L1+L2+L3", "full"]
    
    results = []
    for adv in adversaries:
        for layer in anchor_layers:
            tag = f"{adv} vs {layer}"
            print(f"  {tag}...", end=" ", flush=True)
            r = run_matchup(adv, layer, signals, alpha=1.5)
            results.append(r)
            print(f"SINR={r['sinr_db']:+.2f} dB  wnorm={r['w_norm_final']:.4f}")
    
    return results


if __name__ == "__main__":
    print("=== Nonlinear Adversary Benchmark ===\n")
    np.random.seed(42)
    results = run_full()
    
    Path("nonlinear_results.json").write_text(
        json.dumps(results, indent=2, default=str))
    
    print(f"\nWrote {len(results)} matchup results")
    
    # Summary matrix
    adversaries = ["fxlms", "volterra", "klms", "mlp"]
    anchors = ["baseline", "L1", "L1+L2+L3", "full"]
    
    print(f"\n{'':>12}", end="")
    for a in anchors:
        print(f"{a:>14}", end="")
    print()
    
    for adv in adversaries:
        print(f"{adv:>12}", end="")
        for anc in anchors:
            r = next(x for x in results if x["adversary"] == adv and x["anchor"] == anc)
            print(f"{r['sinr_db']:>+13.2f}", end=" ")
        print()
