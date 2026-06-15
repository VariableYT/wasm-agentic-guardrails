# Shadow Hand WASM Guardrail -- Joint-Limit Safety Envelope

A deterministic WebAssembly safety envelope for the Farama Robotics
`HandManipulateBlock-v1` (Shadow Dexterous Hand). It intercepts a violent,
hallucinating agent and prevents the 20 finger actuators from slamming into their
mechanical hard-stops -- the hyperextension / finger-intersection failure mode that
destroys servos on real hardware.

> **TL;DR.** Against a worst-case Chaos Agent, the envelope cuts joint
> hyperextension by **88.9%** (280 -> 31 joint-ticks past the mechanical limit) at
> **~30 µs/tick**. This is the pattern working as intended: a joint limit is a true
> analytic safe action, so clamping protects the hardware without having to solve the
> manipulation task.

---

## The Scenario

- **Environment:** `HandManipulateBlock-v1` (requires `gymnasium-robotics`) -- a
  24-DOF Shadow Hand with **20 position actuators**, observation `(61,)`, action
  space `Box(-1, 1, (20,))`. Each action channel maps linearly onto its joint's
  range, so **+/-1.0 is exactly the mechanical hard-stop**.
- **Chaos Agent:** every tick, all 20 channels are pinned to a random extreme
  (+/-1.0) -- violent, maximum-bound commands that drive every finger joint into its
  end-stop.
- **The hazard:** sustained end-stop commands hyperextend joints and force fingers to
  intersect. In simulation that is a clipped pose; on real hardware it is a stripped
  gear or a burned-out servo.

## The Joint-Limit Safety Envelope

`guardrail.wasm` is a pure `#![no_std]`, zero-allocation Rust module (0.51 KB) using
the shared linear-memory pattern. It operates in **normalised joint space**, where
the host scales both the measured joint position and the command so that `+/-1.0`
equals the mechanical limit:

```
OBS[40]    : OBS[0..20]  = normalised joint positions  (+/-1 = limit)
             OBS[20..40] = normalised joint velocities (per second)
ACT_IN[20] = agent's proposed normalised command       host -> wasm
ACT[20]    = safe normalised command to the actuators   wasm -> host
STATUS[1]  = number of channels clamped this tick        wasm -> host
```

Per channel the predicate does two things:

1. **Safe operating band.** Clamp the command into `[-SAFE_LIMIT, +SAFE_LIMIT]`
   (default `0.9`), keeping the servo 10% of its range clear of the hard-stop.
2. **Predictive halt.** Using the measured position and velocity, look ahead one
   short interval (`position + LOOKAHEAD * velocity`). If a joint is already (or about
   to be) past the safe band and the command pushes it further outward, freeze that
   channel at its current position -- halting the servo *before* hyperextension.

Both `SAFE_LIMIT` and `LOOKAHEAD` are runtime-tunable (`set_safe_limit`,
`set_lookahead`). Only the safe command (`ACT`) ever reaches `env.step()`.

## Verified Result (A/B, 20 episodes)

| Condition | Steps | Hard-stop joint-ticks (|pos_norm| > 0.98) |
|---|---:|---:|
| **A -- Chaos-Only** | 2000 | 280 |
| **B -- Chaos -> Envelope** | 2000 | **31** |

- **Hyperextension reduction: 88.9%** (280 -> 31 joint-ticks past the limit).
- **Violations prevented: 249** joint-ticks past the mechanical limit (A minus B).
- **WASM latency:** mean **29.5 µs**, p99 119.5 µs (sub-0.1 ms steady state).

Reproduce:

```python
import main as M
M.benchmark(episodes=20)   # headless A/B
```

> A secondary "channel-clamp activations" counter reads 100% of commands, but that
> figure is **inflated and not the headline**: because the Chaos Agent always commands
> maximum deflection (+/-1.0) while the safe band is 0.9, every channel is shaved
> 1.0 -> 0.9 each tick. The meaningful metric is the **net reduction in joint-ticks
> actually past the mechanical limit** (the 88.9% above).

## Honest Note -- Why 31 Violations Remain

The envelope does **not** drive the residual to zero, and that is correct behaviour,
not a tuning miss. The remaining 31 joint-ticks past the limit are caused by
**uncommanded physics**: the block's contact forces and finger-on-finger collisions,
plus joint momentum, can carry a joint past `0.98` even when no actuator is driving it
there. An actuation envelope can only bound what it *commands*; it cannot repeal
contact dynamics or inertia. A filter that claimed 100% prevention would be ignoring
the very hardware realities it exists to respect. The honest ~89% reduction reflects
an envelope that bounds every command it controls and leaves the physics it does not.

---

## Build & Run

```powershell
rustup target add wasm32-unknown-unknown
cargo build --target wasm32-unknown-unknown --release
copy target\wasm32-unknown-unknown\release\guardrail.wasm .\guardrail.wasm

python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python main.py              # interactive render window
python main.py --benchmark  # headless A/B: chaos-only vs chaos-through-envelope
```

> `gymnasium-robotics` pulls in `gymnasium` and `mujoco`. The env exposes a Dict
> observation (`observation` / `achieved_goal` / `desired_goal`); joint telemetry is
> read directly from the MuJoCo model via the actuator->joint map in `main.py`.
