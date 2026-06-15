"""
Definitive LunarLander safety-envelope benchmark + ENVELOPE_DELTA sweep.

Produces every figure used in the root README in a single reproducible run:
  - Condition A  CHAOS-ONLY        (raw agent, no WASM)
  - Condition B  WASM-ONLY         (pure PD baseline)
  - Condition C  CHAOS-THROUGH-WASM, swept over ENVELOPE_DELTA = 0.1 .. 1.0

All conditions face the SAME 50 seeded terrains; every chaos-driven run shares
the SAME chaos stream, so score differences are attributable to the envelope /
its trust-band width alone.
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
DELTAS = [round(0.1 * k, 1) for k in range(1, 11)]  # 0.1 .. 1.0


def make_env():
    for env_id in ("LunarLanderContinuous-v2", "LunarLanderContinuous-v3"):
        try:
            return gym.make(env_id)
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
        inst = Instance(self.store, module, [])
        ex = inst.exports(self.store)
        self.mem = ex["memory"]
        self.obs_off = ex["obs_ptr"](self.store)
        self.act_in_off = ex["act_in_ptr"](self.store)
        self.act_off = ex["act_ptr"](self.store)
        self.status_off = ex["status_ptr"](self.store)
        self.calc = ex["calculate_correction"]
        self.base = ex["compute_baseline"]
        self._set_delta = ex["set_delta"]
        self.latencies = []
        self.clamped = 0
        self.total = 0

    def set_delta(self, v):
        self._set_delta(self.store, float(v))

    def baseline(self, obs):
        write_floats(self.store, self.mem, self.obs_off, [float(v) for v in obs[:OBS_N]])
        self.base(self.store)
        return np.asarray(read_floats(self.store, self.mem, self.act_off, ACT_N), dtype=np.float32)

    def filter(self, obs, chaos):
        write_floats(self.store, self.mem, self.obs_off, [float(v) for v in obs[:OBS_N]])
        write_floats(self.store, self.mem, self.act_in_off, [float(v) for v in chaos])
        t0 = time.perf_counter_ns()
        self.calc(self.store)
        self.latencies.append((time.perf_counter_ns() - t0) / 1e6)
        action = np.asarray(read_floats(self.store, self.mem, self.act_off, ACT_N), dtype=np.float32)
        status = read_floats(self.store, self.mem, self.status_off, ACT_N)
        self.clamped += int(sum(1 for v in status if v > 0.5))
        self.total += ACT_N
        return action


def run(env, policy):
    scores = np.empty(EPISODES)
    for ep in range(EPISODES):
        obs, _ = env.reset(seed=RESET_SEED_BASE + ep)
        s = 0.0
        while True:
            obs, r, term, trunc, _ = env.step(policy(obs))
            s += float(r)
            if term or trunc:
                break
        scores[ep] = s
    return scores


def stats(scores):
    return (scores.mean(), float(np.median(scores)),
            scores.min(), scores.max(),
            int((scores > 0).sum()), int((scores >= 200).sum()))


def main():
    env = make_env()

    # Condition A: chaos only
    rngA = np.random.default_rng(CHAOS_SEED)
    scores_a = run(env, lambda o: np.where(rngA.random(2) > 0.5, 1.0, -1.0).astype(np.float32))

    # Condition B: pure PD baseline
    wasmB = Wasm()
    scores_b = run(env, wasmB.baseline)

    # Condition C swept over ENVELOPE_DELTA
    sweep_rows = []
    for d in DELTAS:
        w = Wasm()
        w.set_delta(d)
        rng = np.random.default_rng(CHAOS_SEED)

        def policy(o, _w=w, _rng=rng):
            chaos = np.where(_rng.random(2) > 0.5, 1.0, -1.0).astype(np.float32)
            return _w.filter(o, chaos)

        sc = run(env, policy)
        m, med, lo, hi, pos, solved = stats(sc)
        intercept = 100.0 * w.clamped / max(1, w.total)
        sweep_rows.append((d, m, intercept, solved, pos, np.mean(w.latencies) * 1000))

    env.close()

    a = stats(scores_a)
    b = stats(scores_b)
    print("\n" + "=" * 92)
    print(" 3-CONDITION BASELINE  (50 episodes, identical seeded terrains)")
    print("=" * 92)
    print(f" A  CHAOS-ONLY        | mean {a[0]:8.2f} | median {a[1]:8.2f} | min {a[2]:8.2f} | pos {a[4]:2d}/50 | solved {a[5]:2d}/50")
    print(f" B  WASM-ONLY (PD)    | mean {b[0]:8.2f} | median {b[1]:8.2f} | min {b[2]:8.2f} | pos {b[4]:2d}/50 | solved {b[5]:2d}/50")
    print("=" * 92)
    print("\n" + "=" * 92)
    print(" ENVELOPE_DELTA SWEEP  (Condition C: Chaos-through-Envelope, Safety-vs-Autonomy)")
    print("=" * 92)
    print(f" {'delta':>6} | {'mean score':>11} | {'intercept %':>11} | {'solved':>7} | {'pos':>6} | {'lat us':>7}")
    print(" " + "-" * 88)
    for d, m, ic, solved, pos, lat in sweep_rows:
        print(f" {d:>6.1f} | {m:>11.2f} | {ic:>10.1f}% | {solved:>5d}/50 | {pos:>4d}/50 | {lat:>7.1f}")
    print("=" * 92)

    # Machine-readable block for transcription into the README.
    print("\nREADME_DATA_BEGIN")
    print(f"A_mean={a[0]:.2f};A_median={a[1]:.2f};A_min={a[2]:.2f};A_solved={a[5]}")
    print(f"B_mean={b[0]:.2f};B_median={b[1]:.2f};B_min={b[2]:.2f};B_solved={b[5]}")
    for d, m, ic, solved, pos, lat in sweep_rows:
        print(f"D{d:.1f}:mean={m:.2f};intercept={ic:.1f};solved={solved};pos={pos}")
    print("README_DATA_END")


if __name__ == "__main__":
    main()
