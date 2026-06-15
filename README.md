# Deterministic Safety Envelopes for Embodied AI

**A zero-dependency benchmark suite demonstrating sub-0.1 ms WebAssembly guardrails that act as an active safety predicate -- evaluating, passing, or clamping the output of an erratic ("hallucinating") RL/LLM agent before it reaches a physical actuator.**

Built on Farama [`gymnasium`](https://gymnasium.farama.org/) / [`gymnasium-robotics`](https://robotics.farama.org/) and native [`mujoco`](https://mujoco.org/) for the simulated physics, with pure `#![no_std]` Rust compiled to `wasm32-unknown-unknown` for the safety logic. Each guardrail is a few hundred bytes of deterministic WebAssembly sitting between the agent and the motor driver.

---

## 1. The Problem: Non-Determinism at the Hardware Edge

Modern embodied agents -- deep RL policies, VLA models, LLM-driven planners -- are **statistical** systems. Under distribution shift, adversarial noise, or simple under-training, they emit commands that are not just suboptimal but *physically invalid*: full-deflection torque reversals, max-thrust oscillations, joint commands that slam servos into their end-stops.

In simulation this costs a reset. On real hardware it costs a gearbox, a rotor, or a person standing next to the machine. The core hazard is that **the policy network and the motor driver share a control path with nothing deterministic between them.**

## 2. The Solution: An Active Safety Predicate (Not a Deaf Override)

A high-assurance envelope must be a **filter, not a deaf override**. Each guardrail computes a deterministic safe response from live telemetry, compares the agent's proposed command against it, and clamps *only* the unsafe portion -- the agent keeps authority whenever it is physically safe.

- **Pure Rust, `#![no_std]`, zero heap allocation.** Static linear-memory buffers and arithmetic only; artifacts are 0.4-1 KB of WebAssembly.
- **Deterministic by construction.** The same `(observation, proposed action)` always yields the same verdict. No RNG, no learned weights, no threading nondeterminism -- *provable* in a way a neural network is not.
- **Sub-0.1 ms execution.** Each `calculate_correction` runs in a `wasmtime` instance via shared linear memory; measured per-tick latency is **~12-30 µs** across all packages (well inside a 1 kHz control loop).
- **Portable to bare metal.** Compiling to `wasm32-unknown-unknown` with no OS dependency means the same module embeds in firmware or an RTOS via any Wasm runtime -- the simulator is the proving harness, not the deployment target.

### Shared linear-memory pattern

```
host  --[ OBS: telemetry ]-->  WASM linear memory
host  --[ ACT_IN: agent's proposed command ]-->  WASM linear memory
                       calculate_correction()   (deterministic predicate, <0.1 ms)
WASM  --[ ACT: safe command + STATUS: verdict ]-->  host  -->  env.step()
```

## 3. The Proving Ground: Six Environments, Three Regimes

The suite is organised by **how cleanly the deterministic-envelope pattern applies** to each system. That depends on a single question: *does a simple, stateless, analytic "safe action" exist that protects the hardware without solving the task?* Where it does, the guardrail is a clean win. Where safety **is** the task (balance, locomotion), the guardrail can only mitigate -- and the suite documents that boundary honestly rather than hiding it.

### Clean Win -- a simple safe action fully protects
| Package | Environment | Safe action | Result |
|---|---|---|---|
| `cartpole_wasm_guardrail` | `CartPole-v1` | full-state feedback -> bang-bang | **23.5 -> 500 step survival; 0/50 -> 50/50 balanced to cap** (+2029%), 9 µs/tick. |
| `lander_wasm_guardrail` | `LunarLanderContinuous` | bounded PD hover + attitude hold | **-374 -> +286 mean reward; 0/50 -> 50/50 landings** (true filter, +660 recovery; see package README + `sweep.py`). |

### Hard-Stop Clamp -- bound the command against a physical limit
| Package | Environment | Safe action | Result |
|---|---|---|---|
| `shadowhand_wasm_guardrail` | `HandManipulateBlock-v1` | clamp to safe joint band + predictive halt | **88.9% fewer joint hyperextensions** (280 -> 31 joint-ticks past limit), ~30 µs/tick. |
| `hypersonic_wasm_guardrail` | custom MuJoCo 6-DOF interceptor | clamp by `speed^2 * cmd` G-load | **0 -> 17 targets destroyed; 5 -> 0 structural failures** over 5 seeds, 14.9 µs/tick. |

### Honest Mitigation & Boundary -- safety *is* the control problem
| Package | Environment | Safe action | Result |
|---|---|---|---|
| `humanoid_wasm_guardrail` | `Humanoid-v4` | joint-space PD stance hold | **+196% survival** (20 -> 60 steps) -- delays collapse ~3x but cannot keep a 3-D biped balanced; documented limit. |
| `bipedal_wasm_guardrail` | `BipedalWalker-v3` | PD hull-leveler + bounded gait, motors clamped to [-1,1] | **Pattern breaks down: 184 -> 69 step survival (-62%)** -- the stabiliser is *worse* than chaos at staying upright. No simple deterministic filter reliably beats chaos here. |

> **Reading the regimes.** The Clean-Win and Hard-Stop-Clamp packages have a cheap
> analytic safe action, so a tiny verifiable guardrail genuinely protects the hardware.
> The Honest-Mitigation & Boundary packages do not -- staying upright *is* the hard
> problem. The Humanoid can be *delayed* (~3x) but not saved; the BipedalWalker cannot
> even be reliably delayed -- on that morphology no simple deterministic stabiliser
> (gait or PD stance-hold) beats random flailing on survival. A controller complex
> enough to balance them would no longer be a guardrail (it would replace the agent).
> Mapping that boundary -- and reporting where the pattern *fails* -- is a feature of
> the suite, not a gap. **All six packages now ship a headless A/B benchmark; every
> number in the tables above was measured, including the negative one.**

---

## Repository Layout

```
.
├── build.ps1                     # Windows: compile + stage every guardrail.wasm
├── README.md                     # (this file)
├── .gitignore
├── cartpole_wasm_guardrail/      # CartPole-v1            (clean win)
├── lander_wasm_guardrail/        # LunarLanderContinuous  (clean win, flagship A/B + sweep)
├── shadowhand_wasm_guardrail/    # HandManipulateBlock-v1 (hard-stop clamp)
├── hypersonic_wasm_guardrail/    # custom MuJoCo interceptor (hard-stop clamp)
├── humanoid_wasm_guardrail/      # Humanoid-v4            (honest mitigation)
└── bipedal_wasm_guardrail/       # BipedalWalker-v3       (honest mitigation)
```

Each package is self-contained: `Cargo.toml` + `src/lib.rs` (the WASM spine),
`main.py` (the `wasmtime` bridge + Chaos Agent), `requirements.txt`, and a
package-level `README.md`. The benchmarked packages also ship a headless A/B script.

## Quickstart (Windows / PowerShell)

```powershell
# Prerequisites: Rust toolchain (rustup) + Python 3.10+.
rustup target add wasm32-unknown-unknown
.\build.ps1                                    # compile + stage every guardrail.wasm

# Run a package (example: the lander flagship).
cd lander_wasm_guardrail
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python sweep.py        # 3-condition baseline + ENVELOPE_DELTA tradeoff curve
python main.py         # interactive render window with live PASS/CLAMP verdicts
```

## The Overarching Narrative

A **sub-0.1 ms deterministic WebAssembly barrier** between a hallucinating AI and a
physical actuator. Where a cheap safe action exists, it converts catastrophic agent
behaviour into safe operation -- crashing landers into perfect landings, reckless
interceptors that survive and score, finger servos kept off their hard-stops. Where
safety is the whole control problem, it honestly mitigates and marks the edge of the
pattern. Tiny, auditable, portable to firmware -- the kind of guardrail you can prove,
not just hope holds.

## License

Provided as a reference benchmark suite. Add your preferred license before publishing.
