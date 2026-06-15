# Bipedal Walker WASM Guardrail

Deterministic WebAssembly safety envelope for `BipedalWalker-v3`. The Chaos Agent
drives all four leg motors to random extreme torque; `guardrail.wasm` intercepts
each tick and substitutes a PD hull-stabilizer + bounded gait oscillator.

> The WASM controller is a stabilizer, not a trained champion walker: it levels
> the hull and clamps every motor command into `[-1, 1]`, demonstrating the
> safety-envelope intercept rather than optimal locomotion.

## Verified benchmark -- a boundary case (A/B, 15 episodes, headless)

```
python benchmark.py
```

| Condition | Hull-survival steps | Episode reward |
|---|---:|---:|
| **A -- Chaos-Only** | 184.3 | -127.6 |
| **B -- WASM stabiliser** | **69.5** | -105.7 |

**Honest result: the envelope does not work here.** On staying upright the WASM
stabiliser is *worse* than random flailing (-62% survival): its open-loop gait commits
the robot to steps that tip it over, whereas chaos twitches in place and stays up
longer. (It earns marginally less-negative reward by making a little forward progress
before falling, but it never gets near solving.)

This is the suite's clearest example of **where the deterministic-envelope pattern
breaks down.** `BipedalWalker` is an underactuated balance problem with no cheap
analytic safe action -- staying upright *is* the control task. We tested the obvious
alternatives too: a tuned joint-space PD stance-hold only reaches ~180-210 steps,
i.e. at or below the noisy chaos baseline (184-255). No simple deterministic filter
reliably beats chaos on this morphology.

Contrast with `CartPole` (a clean win: a 1-DOF balance task with a simple linear safe
action solves to the 500-step cap). The lesson the suite documents: the envelope
pattern protects hardware cheaply where an analytic safe action exists, and degrades
to ineffective where balance itself is the problem. See also `humanoid_wasm_guardrail`,
the same regime, where a PD stance-hold manages a modest ~3x delay but still cannot
keep a 3-D biped standing.

## 1. Compile the Rust payload
```bash
rustup target add wasm32-unknown-unknown
cargo build --target wasm32-unknown-unknown --release
cp target/wasm32-unknown-unknown/release/guardrail.wasm ./guardrail.wasm
```

## 2. Run the bridge
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Requires Box2D (`gymnasium[box2d]`). The console logs chaos vs. WASM torque, the
override magnitude, hull angle, and per-tick WASM latency (target < 0.1 ms).
