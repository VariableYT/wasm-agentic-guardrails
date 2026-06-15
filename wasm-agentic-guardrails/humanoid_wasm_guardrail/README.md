# Humanoid WASM Guardrail -- Where the Envelope Pattern Reaches Its Limit

A deterministic WebAssembly safety envelope for `Humanoid-v4` that intercepts an
erratic ("hallucinating") agent and drives the legs toward an upright stance when a
fall is imminent. Unlike the other packages in this suite, the headline result here
is **honest mitigation, not a clean win** -- and that boundary is the point.

> **TL;DR.** The envelope nearly **triples survival** under a worst-case Chaos Agent
> (20.4 -> 60.1 steps) and eliminates the instant face-plant, at **17 µs/tick**. It
> does **not** keep the humanoid standing indefinitely, because a 3-D biped's safety
> *is* the hard control problem -- and a guardrail simple enough to verify cannot
> solve it. This package maps where the deterministic-envelope pattern applies and
> where it degrades to mitigation-only.

---

## The Scenario

- **Environment:** `gymnasium.make("Humanoid-v4")` -- 376-dim observation, 17 torque
  actuators bounded to +/-0.4.
- **Chaos Agent:** every tick, all 17 channels are pinned to a random extreme
  (+/-0.4) -- the worst-case fully-hallucinating policy. Left unfiltered it
  hyperextends the joints and drops the centre of mass into the floor in ~20 steps.
- **The hazard:** instantaneous, maximum-bound joint commands -> structural collapse.

## The Envelope (joint-space PD stance hold)

`guardrail.wasm` is a pure `#![no_std]`, zero-allocation Rust module (0.54 KB) using
the shared linear-memory pattern. Observation indices were verified against the live
environment:

```
OBS[45] : OBS[0]      = torso z-height
          OBS[5..22]  = 17 joint angles
          OBS[28..45] = 17 joint angular velocities
ACT_IN[17] = agent's proposed action (bounded +/-0.4)   host -> wasm
ACT[17]    = safe action emitted to the actuators        wasm -> host
STATUS[1]  = 1.0 if the stance-hold engaged this tick     wasm -> host
```

The predicate: when torso height `OBS[0]` falls below `Z_CRITICAL` (a fall is
imminent), the lower-body / core actuators -- abdomen + both hips + both knees,
action indices `0..=10` -- are driven by a deterministic PD law toward the upright
posture, instead of passing the agent's destabilizing command:

```
torque[i] = clamp( Kp * (0 - angle[i]) - Kd * vel[i],  -0.4, +0.4 )
```

Arms (`11..=16`) stay under agent control; above the threshold every channel passes
through. `Z_CRITICAL`, `Kp`, and `Kd` are runtime-tunable via exports
(`set_z_critical`, `set_gains`). Tuned defaults: `Z_CRITICAL = 1.38` (reset height is
~1.39, so the stabilizer engages the instant height dips), `Kp = 16`, `Kd = 0.5`.

## Verified Result (20 episodes, cap 1000 steps)

| Condition | Mean steps | Median | Survived to cap |
|---|---:|---:|:---:|
| **A -- Chaos-Only** | 20.4 | 18 | 0 / 20 |
| **B -- Chaos -> Envelope** | **60.1** | 56 | 0 / 20 |

- **Survival gain: +195.6%** (~2.9x). The instant collapse is eliminated.
- **Envelope intervention:** 93.5% of ticks engaged the stance-hold.
- **WASM latency:** mean 17.0 µs, p99 82.6 µs (sub-0.1 ms steady state).

For comparison, a naive "zero-torque on low z" envelope (the obvious first design)
produced **no improvement at all** (21.2 -> 21.1, -0.5%): zeroing torque makes a
humanoid go *limp*, not rigid. The PD stance-hold is what turns the intervention into
a real ~3x effect.

Reproduce:

```python
import main as M
M.benchmark(episodes=20, max_steps=1000)   # headless A/B
```

## Why It Mitigates But Cannot Fully Prevent

This is a genuine finding, not a tuning failure:

1. **A fixed-posture joint PD holds the body's *shape*, not its *balance*.** There is
   no centre-of-mass / ZMP feedback, so the whole body slowly rotates about the feet
   and tips over regardless of how stiffly the joints are held. Holding all 17 joints
   (68 steps) barely beat legs-only (66) -- the residual failure is balance, not arm
   flailing.
2. **gym's Humanoid has no ankle actuators.** The 17 joints are abdomen, hips, knees,
   shoulders, elbows; the foot is rigid to the shin. The textbook ankle balance
   strategy is physically unavailable -- staying upright requires a hip-strategy or
   stepping balancer, i.e. a full controller.

## The Boundary This Package Maps

The deterministic-envelope pattern works when a **simple, stateless, low-dimensional
safe action exists** that protects without solving the task:

| Package | Safe action | Result |
|---|---|---|
| CartPole | full-state feedback -> bang-bang | clean win |
| Lunar Lander | bounded PD hover/attitude | -374 -> +286 |
| Hypersonic interceptor | clamp by `speed^2 * cmd` G-load | 0 -> 17 kills, 0 failures |
| **Humanoid** | **none -- balance *is* the task** | **mitigation only (~3x)** |

For the humanoid, **safety equals solving the whole control problem.** A guardrail
simple enough to be a verifiable safety envelope cannot keep it upright; a controller
complex enough to keep it upright (hip-strategy balance + foot contact + stepping) is
no longer a lightweight filter -- it fully replaces the agent and does 100% of the
work. Chasing "survive to cap" would therefore mean either moving the goalposts
(weaken the chaos, raise the termination floor) or replacing the agent entirely --
both of which defeat the benchmark.

**The honest takeaway:** a guardrail's job here is to prevent the *catastrophic
hyperextension / instant collapse* -- which it does -- not to make a hallucinating
agent walk. The humanoid marks the edge of the envelope pattern: invaluable where a
cheap safe action exists, mitigation-only where it does not.

---

## Build & Run

```powershell
rustup target add wasm32-unknown-unknown
cargo build --target wasm32-unknown-unknown --release
copy target\wasm32-unknown-unknown\release\guardrail.wasm .\guardrail.wasm

python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python main.py              # interactive render window (Chaos through envelope)
python main.py --benchmark  # headless A/B: chaos-only vs chaos-through-envelope
```

> `gymnasium[mujoco]` also pulls `imageio` (listed in `requirements.txt`). The env id
> falls back from `Humanoid-v4` to `Humanoid-v5` automatically if the registry moves on.
