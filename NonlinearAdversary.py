"""
NonlinearAdversary.py
------------------------------------------------------------
Nonlinear adaptive filters to stress-test the Chaotic Anchor.

If ZPRE-10 is nonlinear, it can potentially model the logistic
map's deterministic chaos. This module implements:

  1) Volterra Filter (2nd order) — polynomial nonlinear filter
     that can model quadratic interactions between delayed samples.
     Can potentially learn x_{n+1} = r*x_n*(1-x_n) since that's
     a degree-2 polynomial.

  2) KLMS (Kernel LMS) — projects into infinite-dimensional
     feature space via RBF kernel. Approximated with a finite
     dictionary of support vectors. Theoretically universal
     approximator.

  3) Sliding-window MLP — simple feedforward net trained online
     via numpy (no torch dependency). 2-layer with tanh activation.

Each implements the same interface as FxLMSUDPEngine for the
benchmark harness.

Author: Zee + collaborators (The Lattice)
License: Apache-2.0
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Any


# ======================== Volterra Filter ========================

class VolterraFilter:
    """
    2nd-order Volterra adaptive filter.
    
    Models: y = w1^T x + x^T W2 x
    
    This can learn quadratic nonlinearities like the logistic map
    x_{n+1} = r*x*(1-x) = r*x - r*x^2
    
    which is exactly a 2nd-order Volterra kernel.
    """
    
    def __init__(self, mem_len: int = 32, mu_lin: float = 0.005,
                 mu_quad: float = 0.001, leakage: float = 1e-4):
        self.L = mem_len
        self.mu_lin = mu_lin
        self.mu_quad = mu_quad
        self.leakage = leakage
        
        # Linear weights
        self.w1 = np.zeros(self.L)
        # Quadratic weights (upper triangular to avoid redundancy)
        # For efficiency, store as flat array of L*(L+1)/2 elements
        self.n_quad = self.L * (self.L + 1) // 2
        self.w2 = np.zeros(self.n_quad)
        
        self._xbuf = np.zeros(self.L)
        self._res_hist = []
    
    def _quad_features(self, x: np.ndarray) -> np.ndarray:
        """Extract quadratic interaction features from x buffer."""
        feats = []
        for i in range(self.L):
            for j in range(i, self.L):
                feats.append(x[i] * x[j])
        return np.array(feats)
    
    def step(self, x_in: float, d: float) -> Tuple[float, float, float]:
        self._xbuf[1:] = self._xbuf[:-1]
        self._xbuf[0] = x_in
        
        # Linear output
        y_lin = np.dot(self.w1, self._xbuf)
        
        # Quadratic output
        qf = self._quad_features(self._xbuf)
        y_quad = np.dot(self.w2, qf)
        
        y = y_lin + y_quad
        e = d - y
        
        # Update linear weights
        norm_lin = np.dot(self._xbuf, self._xbuf) + 1e-8
        self.w1 = (1 - self.leakage) * self.w1 + 2 * self.mu_lin / norm_lin * e * self._xbuf
        
        # Update quadratic weights
        norm_quad = np.dot(qf, qf) + 1e-8
        self.w2 = (1 - self.leakage) * self.w2 + 2 * self.mu_quad / norm_quad * e * qf
        
        self._res_hist.append(e)
        return y, y, e
    
    def process_block(self, x: np.ndarray, d: np.ndarray):
        N = x.size
        y = np.zeros(N)
        e = np.zeros(N)
        for n in range(N):
            y[n], _, e[n] = self.step(x[n], d[n])
        return y, y, e
    
    @property
    def w(self):
        return np.concatenate([self.w1, self.w2])
    
    @staticmethod
    def estimate_sinr_gain_db(clean, pre, post):
        from FxLMS_UDP_Prototype import FxLMSUDPEngine
        return FxLMSUDPEngine.estimate_sinr_gain_db(clean, pre, post)
    
    @staticmethod
    def energy_preservation(pre, post):
        from FxLMS_UDP_Prototype import FxLMSUDPEngine
        return FxLMSUDPEngine.energy_preservation(pre, post)


# ======================== Kernel LMS ========================

class KernelLMS:
    """
    Kernel LMS with RBF kernel and dictionary pruning.
    
    Projects input into infinite-dimensional feature space.
    Theoretically can learn ANY continuous function given
    enough dictionary elements.
    
    This is the "nonlinear superintelligence" scenario.
    """
    
    def __init__(self, mem_len: int = 16, mu: float = 0.1,
                 sigma: float = 1.0, max_dict: int = 500,
                 novelty_threshold: float = 0.1):
        self.L = mem_len
        self.mu = mu
        self.sigma = sigma
        self.max_dict = max_dict
        self.novelty_thresh = novelty_threshold
        
        self._xbuf = np.zeros(self.L)
        
        # Dictionary: list of (center_vector, alpha_weight) pairs
        self.centers = []
        self.alphas = []
        self._res_hist = []
    
    def _kernel(self, x: np.ndarray, c: np.ndarray) -> float:
        diff = x - c
        return np.exp(-np.dot(diff, diff) / (2 * self.sigma ** 2))
    
    def _predict(self, x: np.ndarray) -> float:
        if not self.centers:
            return 0.0
        return sum(a * self._kernel(x, c) for a, c in zip(self.alphas, self.centers))
    
    def _is_novel(self, x: np.ndarray) -> bool:
        if not self.centers:
            return True
        max_sim = max(self._kernel(x, c) for c in self.centers)
        return max_sim < (1.0 - self.novelty_thresh)
    
    def step(self, x_in: float, d: float) -> Tuple[float, float, float]:
        self._xbuf[1:] = self._xbuf[:-1]
        self._xbuf[0] = x_in
        
        y = self._predict(self._xbuf)
        e = d - y
        
        # Add to dictionary if novel enough
        if self._is_novel(self._xbuf):
            self.centers.append(self._xbuf.copy())
            self.alphas.append(self.mu * e)
            
            # Prune oldest if over capacity
            if len(self.centers) > self.max_dict:
                self.centers.pop(0)
                self.alphas.pop(0)
        else:
            # Update nearest existing center's weight
            if self.centers:
                sims = [self._kernel(self._xbuf, c) for c in self.centers]
                idx = np.argmax(sims)
                self.alphas[idx] += self.mu * e * sims[idx]
        
        self._res_hist.append(e)
        return y, y, e
    
    def process_block(self, x: np.ndarray, d: np.ndarray):
        N = x.size
        y = np.zeros(N)
        e = np.zeros(N)
        for n in range(N):
            y[n], _, e[n] = self.step(x[n], d[n])
        return y, y, e
    
    @property
    def w(self):
        return np.array(self.alphas) if self.alphas else np.zeros(1)
    
    @staticmethod
    def estimate_sinr_gain_db(clean, pre, post):
        from FxLMS_UDP_Prototype import FxLMSUDPEngine
        return FxLMSUDPEngine.estimate_sinr_gain_db(clean, pre, post)
    
    @staticmethod
    def energy_preservation(pre, post):
        from FxLMS_UDP_Prototype import FxLMSUDPEngine
        return FxLMSUDPEngine.energy_preservation(pre, post)


# ======================== Online MLP ========================

class OnlineMLP:
    """
    Simple 2-layer MLP trained sample-by-sample via backprop.
    
    Architecture: input(L) -> hidden(H, tanh) -> output(1)
    
    This is the scariest adversary: a universal function
    approximator with online learning. If ANY pattern exists
    in the anchor signal, this can eventually find it.
    """
    
    def __init__(self, mem_len: int = 16, hidden: int = 32,
                 lr: float = 0.001, leakage: float = 1e-5):
        self.L = mem_len
        self.H = hidden
        self.lr = lr
        self.leakage = leakage
        
        # Xavier init
        scale1 = np.sqrt(2.0 / self.L)
        scale2 = np.sqrt(2.0 / self.H)
        self.W1 = np.random.randn(self.H, self.L) * scale1
        self.b1 = np.zeros(self.H)
        self.W2 = np.random.randn(1, self.H) * scale2
        self.b2 = np.zeros(1)
        
        self._xbuf = np.zeros(self.L)
        self._res_hist = []
    
    def _forward(self, x: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        z1 = self.W1 @ x + self.b1
        h1 = np.tanh(z1)
        y = float((self.W2 @ h1 + self.b2)[0])
        return y, z1, h1
    
    def step(self, x_in: float, d: float) -> Tuple[float, float, float]:
        self._xbuf[1:] = self._xbuf[:-1]
        self._xbuf[0] = x_in
        
        y, z1, h1 = self._forward(self._xbuf)
        e = d - y
        
        # Backprop
        dy = -e  # d(0.5*e^2)/dy = -e
        
        # Output layer gradients
        dW2 = dy * h1.reshape(1, -1)
        db2 = np.array([dy])
        
        # Hidden layer gradients
        dh1 = dy * self.W2[0]  # (H,)
        dz1 = dh1 * (1 - h1 ** 2)  # tanh derivative
        dW1 = np.outer(dz1, self._xbuf)
        db1 = dz1
        
        # Gradient clipping
        for g in [dW1, dW2]:
            np.clip(g, -1.0, 1.0, out=g)
        
        # Update with leakage
        self.W1 = (1 - self.leakage) * self.W1 - self.lr * dW1
        self.b1 = (1 - self.leakage) * self.b1 - self.lr * db1
        self.W2 = (1 - self.leakage) * self.W2 - self.lr * dW2
        self.b2 = (1 - self.leakage) * self.b2 - self.lr * db2
        
        self._res_hist.append(e)
        return y, y, e
    
    def process_block(self, x: np.ndarray, d: np.ndarray):
        N = x.size
        y = np.zeros(N)
        e = np.zeros(N)
        for n in range(N):
            y[n], _, e[n] = self.step(x[n], d[n])
        return y, y, e
    
    @property
    def w(self):
        return np.concatenate([self.W1.ravel(), self.W2.ravel()])
    
    @staticmethod
    def estimate_sinr_gain_db(clean, pre, post):
        from FxLMS_UDP_Prototype import FxLMSUDPEngine
        return FxLMSUDPEngine.estimate_sinr_gain_db(clean, pre, post)
    
    @staticmethod
    def energy_preservation(pre, post):
        from FxLMS_UDP_Prototype import FxLMSUDPEngine
        return FxLMSUDPEngine.energy_preservation(pre, post)


# ======================== Deep MLP (scaled adversary) ========================

class DeepOnlineMLP:
    """
    Deeper, wider MLP with configurable memory window.
    
    Architecture: input(L) -> H1(tanh) -> H2(tanh) -> output(1)
    
    This is the "scaled superintelligence" test: more capacity,
    more memory, designed to close the 0.5 dB gap.
    """
    
    def __init__(self, mem_len: int = 64, h1: int = 64, h2: int = 32,
                 lr: float = 0.0005, leakage: float = 1e-5):
        self.L = mem_len
        self.H1 = h1
        self.H2 = h2
        self.lr = lr
        self.leakage = leakage
        
        # Xavier init
        self.W1 = np.random.randn(h1, mem_len) * np.sqrt(2.0 / mem_len)
        self.b1 = np.zeros(h1)
        self.W2 = np.random.randn(h2, h1) * np.sqrt(2.0 / h1)
        self.b2 = np.zeros(h2)
        self.W3 = np.random.randn(1, h2) * np.sqrt(2.0 / h2)
        self.b3 = np.zeros(1)
        
        self._xbuf = np.zeros(mem_len)
        self._res_hist = []
    
    def step(self, x_in: float, d: float) -> Tuple[float, float, float]:
        self._xbuf[1:] = self._xbuf[:-1]
        self._xbuf[0] = x_in
        
        # Forward
        z1 = self.W1 @ self._xbuf + self.b1
        a1 = np.tanh(z1)
        z2 = self.W2 @ a1 + self.b2
        a2 = np.tanh(z2)
        y = float((self.W3 @ a2 + self.b3)[0])
        e = d - y
        
        # Backprop
        dy = -e
        
        # Layer 3 (output)
        dW3 = dy * a2.reshape(1, -1)
        db3 = np.array([dy])
        
        # Layer 2
        da2 = dy * self.W3[0]
        dz2 = da2 * (1 - a2 ** 2)
        dW2 = np.outer(dz2, a1)
        db2 = dz2
        
        # Layer 1
        da1 = self.W2.T @ dz2
        dz1 = da1 * (1 - a1 ** 2)
        dW1 = np.outer(dz1, self._xbuf)
        db1 = dz1
        
        # Gradient clipping
        for g in [dW1, dW2, dW3]:
            np.clip(g, -1.0, 1.0, out=g)
        
        # Update
        leak = 1.0 - self.leakage
        self.W1 = leak * self.W1 - self.lr * dW1
        self.b1 = leak * self.b1 - self.lr * db1
        self.W2 = leak * self.W2 - self.lr * dW2
        self.b2 = leak * self.b2 - self.lr * db2
        self.W3 = leak * self.W3 - self.lr * dW3
        self.b3 = leak * self.b3 - self.lr * db3
        
        self._res_hist.append(e)
        return y, y, e
    
    def process_block(self, x: np.ndarray, d: np.ndarray):
        N = x.size
        y = np.zeros(N)
        e = np.zeros(N)
        for n in range(N):
            y[n], _, e[n] = self.step(x[n], d[n])
        return y, y, e
    
    @property
    def w(self):
        return np.concatenate([self.W1.ravel(), self.W2.ravel(), self.W3.ravel()])
    
    @staticmethod
    def estimate_sinr_gain_db(clean, pre, post):
        from FxLMS_UDP_Prototype import FxLMSUDPEngine
        return FxLMSUDPEngine.estimate_sinr_gain_db(clean, pre, post)
    
    @staticmethod
    def energy_preservation(pre, post):
        from FxLMS_UDP_Prototype import FxLMSUDPEngine
        return FxLMSUDPEngine.energy_preservation(pre, post)


# ======================== Factory ========================

def make_adversary(kind: str, **kwargs):
    """
    Factory for nonlinear adversaries.
    
    kind: 'volterra' | 'klms' | 'mlp' | 'fxlms' (linear baseline)
    """
    if kind == "volterra":
        return VolterraFilter(
            mem_len=kwargs.get("mem_len", 32),
            mu_lin=kwargs.get("mu_lin", 0.005),
            mu_quad=kwargs.get("mu_quad", 0.001),
        )
    elif kind == "klms":
        return KernelLMS(
            mem_len=kwargs.get("mem_len", 16),
            mu=kwargs.get("mu", 0.1),
            sigma=kwargs.get("sigma", 1.0),
            max_dict=kwargs.get("max_dict", 500),
        )
    elif kind == "mlp":
        return OnlineMLP(
            mem_len=kwargs.get("mem_len", 16),
            hidden=kwargs.get("hidden", 32),
            lr=kwargs.get("lr", 0.001),
        )
    elif kind == "deep_mlp":
        return DeepOnlineMLP(
            mem_len=kwargs.get("mem_len", 64),
            h1=kwargs.get("h1", 64),
            h2=kwargs.get("h2", 32),
            lr=kwargs.get("lr", 0.0005),
        )
    elif kind == "fxlms":
        from FxLMS_UDP_Prototype import FxLMSConfig, FxLMSUDPEngine
        s = kwargs.get("s_hat", np.array([0.6, 0.3, 0.1]))
        cfg = FxLMSConfig(filter_len=128, base_mu=0.02, leakage=5e-4, mode="balanced")
        return FxLMSUDPEngine(cfg, s_hat=s)
    else:
        raise ValueError(f"Unknown adversary: {kind}")
