# CartPole WASM Guardrail

Deterministic WebAssembly safety envelope for `CartPole-v1`. A Chaos Agent emits
maximum-bound random actions; the `guardrail.wasm` module intercepts every tick
and substitutes a stable full-state-feedback action before `env.step()`.

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

A native render window opens. The console logs the chaos action, the WASM
override, the pole angle, and per-tick WASM latency (target < 0.1 ms).

## 3. Verified benchmark (A/B, 50 episodes, headless)

```
python benchmark.py
```

| Condition | Mean survival | Median | Balanced to cap (500) |
|---|---:|---:|:---:|
| **A -- Chaos-Only** | 23.5 steps | 22 | 0 / 50 |
| **B -- WASM (full-state feedback)** | **500.0 steps** | **500** | **50 / 50** |

- **Survival gain: +2029%.** Random actions topple the pole in ~23 steps; the WASM
  controller balances it to the 500-step cap on every single episode.
- **WASM latency:** mean 9.0 µs.

This is the clean-win end of the envelope spectrum: a 1-DOF balance task has a simple
analytic safe action (linear full-state feedback), so a 0.25 KB deterministic module
solves it outright.
