"""
BoundaryProbe.py — The decisive experiment.
Maps the boundary where MLP absorbs the anchor.
Tests both sides: adversary capacity vs discontinuity rate.
"""
import json, numpy as np
from pathlib import Path
from typing import Dict, Any
from FxLMS_UDP_Prototype import FxLMSConfig, FxLMSUDPEngine
from ChaoticAnchor import AnchorConfig, ChaoticAnchor

class DeepMLP:
    def __init__(self, mem_len=16, hidden=64, depth=2, lr=0.0008, leakage=1e-5):
        self.L, self.lr, self.leakage, self.depth = mem_len, lr, leakage, depth
        self.weights, self.biases = [], []
        in_dim = mem_len
        for _ in range(depth):
            self.weights.append(np.random.randn(hidden, in_dim)*np.sqrt(2.0/in_dim))
            self.biases.append(np.zeros(hidden)); in_dim = hidden
        self.weights.append(np.random.randn(1, hidden)*np.sqrt(2.0/hidden))
        self.biases.append(np.zeros(1))
        self._xbuf = np.zeros(mem_len)

    def step(self, x_in, d):
        self._xbuf[1:] = self._xbuf[:-1]; self._xbuf[0] = x_in
        acts, pres = [self._xbuf.copy()], []
        x = self._xbuf.copy()
        for i in range(self.depth):
            z = self.weights[i] @ x + self.biases[i]; pres.append(z)
            x = np.tanh(z); acts.append(x)
        y = float((self.weights[-1] @ x + self.biases[-1])[0]); e = d - y
        delta = np.array([-e])
        dW = np.outer(delta, acts[-1]); np.clip(dW, -1, 1, out=dW)
        self.weights[-1] = (1-self.leakage)*self.weights[-1] - self.lr*dW
        self.biases[-1] = (1-self.leakage)*self.biases[-1] - self.lr*delta
        delta = (self.weights[-1].T @ delta).ravel()
        for i in range(self.depth-1, -1, -1):
            delta = delta * (1 - np.tanh(pres[i])**2)
            dW = np.outer(delta, acts[i]); np.clip(dW, -1, 1, out=dW)
            self.weights[i] = (1-self.leakage)*self.weights[i] - self.lr*dW
            self.biases[i] = (1-self.leakage)*self.biases[i] - self.lr*delta
            if i > 0: delta = (self.weights[i].T @ delta)
        return y, y, e

    @property
    def w(self): return np.concatenate([w.ravel() for w in self.weights])

def make_anchor_tuned(disc_rate="normal", alpha=1.5):
    cfg = AnchorConfig(alpha=alpha, enable_chaos=True, enable_adaptive=True,
                       enable_private_key=True, enable_orthogonal=True)
    rates = {"slow":(1000,0.01,100),"normal":(500,0.02,50),"fast":(200,0.05,25),"extreme":(80,0.10,10)}
    cfg.key_transition_interval, cfg.orthogonal_switch_prob, cfg.tau_samples = rates[disc_rate]
    return ChaoticAnchor(cfg)

def gen_scene(N=30000, seed=42):
    rng = np.random.default_rng(seed); t = np.arange(N)/16000.0
    clean = 0.5*np.sin(2*np.pi*200*t)
    v = rng.normal(0,1,size=N); x = np.convolve(v, [1.0,-0.65,0.3], mode="same")
    d0 = clean + np.convolve(x, [0.8,0.2,0.05], mode="same")
    return {"clean": clean, "x": x, "d0": d0}

def run_one(mlp_kw, disc_rate, alpha, signals, N):
    x0,d0,clean = signals["x"][:N], signals["d0"][:N], signals["clean"][:N]
    np.random.seed(42); mlp = DeepMLP(**mlp_kw)
    anchor = make_anchor_tuned(disc_rate, alpha)
    e_out = np.zeros(N)
    for n in range(N):
        a = anchor.generate_sample(n, error=(e_out[n-1] if n>0 else 0.0))
        _,_,e_out[n] = mlp.step(x0[n]+0.6*a, d0[n]+a)
    sinr = FxLMSUDPEngine.estimate_sinr_gain_db(clean, pre=d0, post=e_out)
    W = 1000; conv = []
    for i in range(W, N, W//2):
        s = FxLMSUDPEngine.estimate_sinr_gain_db(clean[i-W:i], pre=d0[i-W:i], post=e_out[i-W:i])
        conv.append({"t":int(i),"s":round(float(s),3)})
    ls = int(0.75*N)
    late = FxLMSUDPEngine.estimate_sinr_gain_db(clean[ls:], pre=d0[ls:], post=e_out[ls:])
    return {"sinr":round(float(sinr),3),"late":round(float(late),3),"conv":conv,
            "params":sum(w.size for w in mlp.weights)}

def run_baseline(mlp_kw, signals, N):
    x0,d0,clean = signals["x"][:N], signals["d0"][:N], signals["clean"][:N]
    np.random.seed(42); mlp = DeepMLP(**mlp_kw)
    e = np.zeros(N)
    for n in range(N): _,_,e[n] = mlp.step(x0[n], d0[n])
    return round(float(FxLMSUDPEngine.estimate_sinr_gain_db(clean, pre=d0, post=e)),3)

def main():
    N = 30000; signals = gen_scene(N=N); R = {}

    print("=== Exp 1: Adversary Scaling ===")
    cfgs = [
        {"mem_len":16,"hidden":32,"depth":1,"lr":0.001},
        {"mem_len":32,"hidden":64,"depth":1,"lr":0.0008},
        {"mem_len":64,"hidden":64,"depth":2,"lr":0.0005},
        {"mem_len":64,"hidden":128,"depth":2,"lr":0.0005},
        {"mem_len":128,"hidden":128,"depth":3,"lr":0.0003},
        {"mem_len":256,"hidden":64,"depth":2,"lr":0.0005},
    ]
    exp1 = []
    for c in cfgs:
        lab = f"m{c['mem_len']}/h{c['hidden']}/d{c['depth']}"
        print(f"  {lab}...", end=" ", flush=True)
        r = run_one(c, "normal", 1.5, signals, N); r["label"]=lab; r["cfg"]=c
        exp1.append(r); print(f"SINR={r['sinr']:+.3f} late={r['late']:+.3f} p={r['params']}")
    R["adv_scale"] = exp1

    print("\n=== Exp 2: Discontinuity Scaling ===")
    fixed = {"mem_len":64,"hidden":128,"depth":2,"lr":0.0005}
    exp2 = []
    for dr in ["slow","normal","fast","extreme"]:
        print(f"  {dr}...", end=" ", flush=True)
        r = run_one(fixed, dr, 1.5, signals, N); r["dr"]=dr; exp2.append(r)
        print(f"SINR={r['sinr']:+.3f} late={r['late']:+.3f}")
    R["disc_scale"] = exp2

    print("\n=== Exp 3: Race Matrix ===")
    sm = {"mem_len":16,"hidden":32,"depth":1,"lr":0.001}
    bg = {"mem_len":128,"hidden":128,"depth":3,"lr":0.0003}
    exp3 = []
    for al,ac in [("small",sm),("big",bg)]:
        for dr in ["slow","normal","fast","extreme"]:
            print(f"  {al} vs {dr}...", end=" ", flush=True)
            r = run_one(ac, dr, 1.5, signals, N)
            r["adv"]=al; r["dr"]=dr; exp3.append(r)
            print(f"late={r['late']:+.3f}")
    R["race"] = exp3

    print("\n=== Baselines ===")
    for al,ac in [("small",sm),("big",bg)]:
        b = run_baseline(ac, signals, N); print(f"  {al}: {b:+.3f}")
    R["baselines"] = {"small": run_baseline(sm, signals, N), "big": run_baseline(bg, signals, N)}

    Path("boundary_results.json").write_text(json.dumps(R, indent=2))
    print(f"\nDone.")

if __name__=="__main__": main()
