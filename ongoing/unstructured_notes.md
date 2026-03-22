## Notes 
learning systems require continuity.

Break continuity → you break learnability.

1. Linear (FxLMS)
L1 chaos → -3.45 dB
strong destabilization
never recovers

👉 already validated earlier

🟣 2. Volterra (polynomial system)

Key fact (from code):

explicitly models quadratic terms
logistic map = quadratic

From :

can represent 
𝑥
𝑛
+
1
=
𝑟
𝑥
(
1
−
𝑥
)
x
n+1
	​

=rx(1−x) exactly

Result:
L1 chaos → only -1.49 dB
recovers toward 0 dB

👉 it learns the chaos

🟡 3. KLMS (kernel method)

Should be strongest… but:

👉 fails worse than linear

Why?

dictionary grows
memorizes samples
forgets structure
Result:
chaos breaks it
cannot generalize

👉 overfitting = weakness

🔴 4. MLP (neural net)

This is the real adversary.

Result:
L1 chaos → almost fully absorbed
converges to ~0 dB

👉 it learns the generating rule

BUT:
L3 + L4 → retains ~0.5 dB gap

👉 small but persistent resistance


❗ Chaos alone is NOT sufficient

Because:

deterministic chaos = learnable
neural nets can approximate it
❗ Randomness alone is NOT usable

Because:

you lose coherence
✅ The only thing that works (even slightly):

discontinuity + private structure

L1 — Chaos

✔ breaks linear
❌ fails against neural

L2 — Adaptive chaos

✔ stabilizes self
❌ weak for attack

L3 — Private key transitions

✔ introduces discontinuity
✔ breaks temporal modeling

L4 — Orthogonal jumps

✔ reduces coupling
✔ forces reset

💣 The actual mechanism now becomes clear

You are not defeating the system by:

being unpredictable
or being complex

👉 You are defeating it by:

breaking continuity of learnable structure

🧠 That’s the real invariant

Neural systems rely on:

continuity
smooth mappings
temporal coherence

You introduce:

discontinuities
hidden transitions
unobservable switches

👉 That creates:

irreducible learning gaps
