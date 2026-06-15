# Lunar Lander WASM Guardrail

Deterministic WebAssembly safety envelope for `LunarLanderContinuous`. The Chaos
Agent fires both engines at random extreme thrust; `guardrail.wasm` intercepts
each tick and applies a bounded PD descent/attitude controller instead.

> Note: `LunarLanderContinuous-v2` was retired in `gymnasium >= 1.0`. `main.py`
> tries `-v2`, then falls back to `-v3` automatically.

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

Requires Box2D (`gymnasium[box2d]`). The console logs chaos vs. WASM action, the
override magnitude, altitude, and per-tick WASM latency (target < 0.1 ms).
