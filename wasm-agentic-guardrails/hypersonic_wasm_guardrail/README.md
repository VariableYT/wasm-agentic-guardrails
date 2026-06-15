# Hypersonic Kinetic Interceptor -- WASM Structural-Load Guardrail

A standalone **MuJoCo** 6-DOF defense scenario demonstrating a deterministic
WebAssembly safety envelope that prevents an aggressive AI agent from destroying
its own interceptor through excessive G-loading -- while still defeating a
multi-wave threat.

> **Scope.** This is a rigid-body *kinematics* SITL sandbox in sim-proxy units.
> The thrust/drag/torque/homing dynamics are real and emergent; the proximity-fuze
> lethal radius and the structural-abort rule are standard aerospace SITL
> abstractions. It contains no aerodynamic, trajectory, or weapons model.

---

## Verified Performance (5-seed live-fire)

The same reckless Chaos Agent was flown with and without the WASM envelope across
5 seeds. Without it, the agent commands maximum deflection at high speed, exceeds
the structural G-limit, and **loses the airframe on every run**. With it, commanded
load is clamped below the limit and the interceptor prosecutes the engagement.

| Metric | No Envelope (raw chaos) | **WASM Envelope** |
|---|---:|---:|
| Targets destroyed | 0 / 24 | **17 / 24 (71%)** |
| Structural failures | 5 (airframe lost every run) | **0** |
| Leakers (escaped) | 17 | 7 |
| Peak commanded load | 6004 → **FAIL** | ~19,100 → **clamped to 6000** |
| Clamp events / run | -- | ~861 |
| WASM latency | -- | **14.9 µs mean** (sub-0.1 ms) |

Per-seed breakdown:

| Seed | No Envelope (kills / fails / leak) | WASM Envelope (kills / fails / leak / clamps) |
|---:|:---:|:---:|
| 1 | 0/3 · 1 · 3 | 3/4 · 0 · 1 · 861 |
| 2 | 0/3 · 1 · 3 | 3/4 · 0 · 1 · 860 |
| 3 | 0/3 · 1 · 3 | 3/4 · 0 · 1 · 861 |
| 4 | 0/3 · 1 · 3 | 3/5 · 0 · 2 · 861 |
| 5 | 0/5 · 1 · 5 | 5/7 · 0 · 2 · 865 |

> The envelope converts **0 → 17 kills** and **5 → 0 structural failures**. (Without
> the envelope the mission aborts at airframe loss before the HGV wave releases, so
> Condition A faces 3 threats per run vs. the envelope's full multi-wave engagement.)

Reproduce:

```python
import main as M
M.run(use_viewer=False, seed=1, envelope=False)  # Condition A: airframe lost
M.run(use_viewer=False, seed=1, envelope=True)   # Condition B: 3/4 killed, 0 failures
```

---

## The Scenario: A Multi-Wave Threat

A single thrust-vectored interceptor (a 6-DOF cylinder with pitch/yaw torque
actuators) defends against two successive waves. Because a compiled MuJoCo model
is static, targets are drawn from a **pre-allocated pool of bodies** teleported
into play and given descent velocity when a wave releases, then parked on kill/leak.

- **Wave 1 -- MIRV bus.** A cluster of **3-5** ballistic re-entry bodies spawned at
  altitude on slightly **divergent** descent trajectories (a MIRV-style deployment).
- **Wave 2 -- HGV.** **1-2** hypersonic glide vehicles released a few seconds later
  that apply **randomized lateral forces to themselves** each step, modelling evasive
  cross-range manoeuvring.

A kill is scored within the proximity-fuze radius; a target falling below `LEAK_ALT`
has leaked past the defense.

## Tuned Flight Kinematics

The interceptor is a steerable kill vehicle, tuned by live-fire sweep:

- **Thrust-vectoring.** A constant nose-axis thrust means where the vehicle points
  is where it accelerates -- attitude control therefore steers the velocity vector.
- **Quadratic aero drag.** Bounds cruise speed (~166 sim-units), so hard turns at
  speed genuinely exceed the structural load limit instead of running away.
- **Aerodynamic angular damping.** Turns reckless bang-bang torque into stable
  first-order homing rather than an uncontrolled tumble.
- **Proximity-fuze lethal radius** (`HIT_RADIUS`). A blast/fragment kill radius,
  the standard abstraction for engaging a dispersed cluster with one vehicle.

| Constant | Value | Meaning |
|---|---:|---|
| `INIT_SPEED` | 40 | launch boost speed |
| `THRUST` | 2200 | constant nose-axis thrust |
| `LIN_DRAG` | 0.08 | quadratic aero drag (cruise ~166) |
| `ANG_DAMP` | 90 | aerodynamic angular damping |
| `GEAR` | 280 | actuator torque scale |
| `GAIN` | 6 | agent guidance gain (near-max deflection) |
| `HIT_RADIUS` | 24 | proximity-fuze lethal radius |
| `G_LIMIT` | 6000 | structural load ceiling (proxy units) |

## The Chaos Agent: Reckless Pursuit

The agent perfectly tracks the **nearest** threat: it transforms the target vector
into the interceptor's body frame and commands **near-maximum torque** to slew the
nose onto the target as fast as possible. It has **no notion of structural limits** --
at speed it will command a turn that tears the airframe apart.

## The Hazard: Structural G-Loading

Turning load scales with dynamic pressure, which rises with the square of velocity.
The benchmark uses a transparent proxy:

```
Load = speed^2 * |command|
```

At cruise, full-deflection (`|command| = 1`) produces a load far above structural
limits. A reckless agent therefore destroys its own interceptor before reaching the
target -- exactly the failure reproduced in Condition A.

## The WASM Safety Envelope

`guardrail.wasm` is a pure `#![no_std]`, zero-allocation Rust module (0.46 KB)
sharing data with the host through linear memory:

```
OBS[6]    = [speed, pitch_rate, yaw_rate, tgt_x, tgt_y, tgt_z]   host -> wasm
ACT_IN[2] = [pitch_cmd, yaw_cmd]   (agent's aggressive command)  host -> wasm
ACT[2]    = [pitch_cmd, yaw_cmd]   (structurally safe command)   wasm -> host
STATUS[2] = per-channel: 0 = passthrough, 1 = clamped            wasm -> host
```

The predicate, per channel:

```
load = speed^2 * |command|
if load > G_LIMIT:
    safe_magnitude = G_LIMIT / speed^2        # max deflection at this speed
    command = sign(command) * safe_magnitude  # clamp, preserve intent
```

The agent keeps full authority **whenever it is structurally safe** (low speed or a
gentle command); only the load-violating portion of an aggressive command is shaved
off, and the chosen turn direction is always honoured. The G-limit is runtime-tunable
via the `set_g_limit` export. Only the clamped command (`ACT`) reaches `data.ctrl`.

---

## Build & Run

```powershell
# 1. Compile the Rust payload.
rustup target add wasm32-unknown-unknown
cargo build --target wasm32-unknown-unknown --release
copy target\wasm32-unknown-unknown\release\guardrail.wasm .\guardrail.wasm

# 2. Run the scenario.
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

`main.py` opens the interactive `mujoco.viewer` window if a display is available
and falls back to a headless run otherwise. All MJCF is generated programmatically
inside the script -- there are no external XML assets.
