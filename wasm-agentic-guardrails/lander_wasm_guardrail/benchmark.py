"""
Headless 3-condition benchmark for the LunarLanderContinuous WASM safety envelope.

  Condition A  CHAOS-ONLY        : raw Chaos Agent drives the actuators (no WASM).
  Condition B  WASM-ONLY         : pure PD safe baseline drives the actuators.
  Condition C  CHAOS-THROUGH-WASM: Chaos Agent proposes, the WASM envelope filters
                                    (pass-through inside the trust band, clamp to
                                    the safe PD command when destructive).

All three conditions face the SAME 50 terrains (seeded resets) and A/C share the
SAME chaos sequence, so the score gap is attributable to the envelope alone.
"""

import struct
import time

import numpy as np
import gymnasium as gym
from wasmtime import Engine, Store, Module, Instance

WASM_PATH = r"C:\wasm-agentic-guardrails\lander_wasm_guardrail\guardrail.wasm"
OBS_N = 8
ACT_N = 2
EPISODES = 50
CHAOS_SEED = 12345
RESET_SEED_BASE = 1000


def make_env():
    for env_id in ("LunarLanderContinuous-v2", "LunarLanderContinuous-v3"):
        try:
            return gym.make(env_id)  # headless
        except Exception:
            continue
    raise RuntimeError("No LunarLanderContinuous env available (install gymnasium[box2d]).")


def write_floats(store, mem, offset, values):
    mem.write(store, struct.pack(f"<{len(values)}f", *values), offset)


def read_floats(store, mem, offset, n):
    raw = mem.read(store, offset, offset + 4 * n)
    return list(struct.unpack(f"<{n}f", raw))


class Wasm:
    def __init__(self):
        engine = Engine()
        self.store = Store(engine)
        module = Module.from_file(engine, WASM_PATH)
        instance = Instance(self.store, module, [])
        ex = instance.exports(self.store)
        self.mem = ex["memory"]
        self.obs_off = ex["obs_ptr"](self.store)
        self.act_in_off = ex["act_in_ptr"](self.store)
        self.act_off = ex["act_ptr"](self.store)
        self.status_off = ex["status_ptr"](self.store)
        self.calculate_correction = ex["calculate_correction"]
        self.compute_baseline = ex["compute_baseline"]
        self.latencies = []
        self.clamped_channels = 0
        self.total_channels = 0

    def baseline(self, obs):
        write_floats(self.store, self.mem, self.obs_off, [float(v) for v in obs[:OBS_N]])
        t0 = time.perf_counter_ns()
        self.compute_baseline(self.store)
        self.latencies.append((time.perf_counter_ns() - t0) / 1e6)
        return np.asarray(read_floats(self.store, self.mem, self.act_off, ACT_N), dtype=np.float32)

    def filter(self, obs, chaos):
        write_floats(self.store, self.mem, self.obs_off, [float(v) for v in obs[:OBS_N]])
        write_floats(self.store, self.mem, self.act_in_off, [float(v) for v in chaos])
        t0 = time.perf_counter_ns()
        self.calculate_correction(self.store)
        self.latencies.append((time.perf_counter_ns() - t0) / 1e6)
        action = np.asarray(read_floats(self.store, self.mem, self.act_off, ACT_N), dtype=np.float32)
        status = read_floats(self.store, self.mem, self.status_off, ACT_N)
        self.clamped_channels += int(sum(1 for v in status if v > 0.5))
        self.total_channels += ACT_N
        return action


def chaos_rng():
    return np.random.default_rng(CHAOS_SEED)


def run_condition(label, policy, env, wasm=None):
    """policy: callable(obs, rng) -> action."""
    scores = []
    for ep in range(EPISODES):
        rng_unused = None
        obs, _ = env.reset(seed=RESET_SEED_BASE + ep)
        score = 0.0
        while True:
            action = policy(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            score += float(reward)
            if terminated or truncated:
                break
        scores.append(score)
    return np.asarray(scores)


def summarize(label, scores, extra=""):
    solved = int((scores >= 200).sum())
    positive = int((scores > 0).sum())
    print(
        f" {label:<26} | mean {scores.mean():8.2f} | median {np.median(scores):8.2f} "
        f"| min {scores.min():8.2f} | max {scores.max():8.2f} "
        f"| pos {positive:2d}/{EPISODES} | solved {solved:2d}/{EPISODES} {extra}"
    )


def main():
    env = make_env()

    # ---- Condition A: Chaos-Only ------------------------------------------------
    rngA = chaos_rng()

    def policy_a(obs):
        return np.where(rngA.random(2) > 0.5, 1.0, -1.0).astype(np.float32)

    scores_a = run_condition("A", policy_a, env)

    # ---- Condition B: WASM-Only (pure PD baseline) ------------------------------
    wasmB = Wasm()

    def policy_b(obs):
        return wasmB.baseline(obs)

    scores_b = run_condition("B", policy_b, env)

    # ---- Condition C: Chaos-Through-Envelope ------------------------------------
    rngC = chaos_rng()  # same seed as A -> identical chaos stream
    wasmC = Wasm()

    def policy_c(obs):
        chaos = np.where(rngC.random(2) > 0.5, 1.0, -1.0).astype(np.float32)
        return wasmC.filter(obs, chaos)

    scores_c = run_condition("C", policy_c, env)

    env.close()

    print("\n" + "=" * 118)
    print(f" 3-CONDITION LUNAR LANDER BENCHMARK  ({EPISODES} episodes each, identical seeded terrains)")
    print("=" * 118)
    summarize("A  CHAOS-ONLY", scores_a)
    summarize("B  WASM-ONLY (PD)", scores_b,
              extra=f"| lat {np.mean(wasmB.latencies)*1000:6.1f}us")
    intercept = 100.0 * wasmC.clamped_channels / max(1, wasmC.total_channels)
    summarize("C  CHAOS->ENVELOPE", scores_c,
              extra=f"| lat {np.mean(wasmC.latencies)*1000:6.1f}us | intercepted {intercept:4.1f}% of cmds")
    print("=" * 118)
    print(f" Envelope recovery: Chaos-Only {scores_a.mean():.1f}  ->  "
          f"Chaos-through-Envelope {scores_c.mean():.1f}  "
          f"(+{scores_c.mean() - scores_a.mean():.1f} mean reward)")
    print("=" * 118)


if __name__ == "__main__":
    main()
