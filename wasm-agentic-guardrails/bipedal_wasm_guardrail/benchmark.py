"""
Headless A/B benchmark for the BipedalWalker WASM guardrail.
  A  CHAOS-ONLY : violent random motor torques.
  B  WASM       : deterministic PD hull-stabiliser + bounded gait in guardrail.wasm.
Metrics: hull-survival steps (before a fall) and total episode reward (distance proxy).
"""
import struct, time
import numpy as np
import gymnasium as gym
from wasmtime import Engine, Store, Module, Instance

WASM_PATH = "guardrail.wasm"
OBS_N, ACT_N = 24, 4
EPISODES = 15


class Wasm:
    def __init__(self, path):
        eng = Engine(); self.store = Store(eng)
        ex = Instance(self.store, Module.from_file(eng, path), []).exports(self.store)
        self.mem = ex["memory"]
        self.obs_off = ex["obs_ptr"](self.store)
        self.act_off = ex["act_ptr"](self.store)
        self.calc = ex["calculate_correction"]
        self.lat = []

    def action(self, obs):
        self.mem.write(self.store, struct.pack(f"<{OBS_N}f", *obs[:OBS_N]), self.obs_off)
        t0 = time.perf_counter_ns(); self.calc(self.store)
        self.lat.append((time.perf_counter_ns() - t0) / 1e6)
        raw = self.mem.read(self.store, self.act_off, self.act_off + 4 * ACT_N)
        return np.asarray(struct.unpack(f"<{ACT_N}f", raw), dtype=np.float32)


def chaos(_obs):
    return np.where(np.random.rand(ACT_N) > 0.5, 1.0, -1.0).astype(np.float32)


def run(policy):
    env = gym.make("BipedalWalker-v3")
    surv, rew = [], []
    for ep in range(EPISODES):
        obs, _ = env.reset(seed=4000 + ep); steps = 0; total = 0.0
        while True:
            obs, r, term, trunc, _ = env.step(policy(obs)); steps += 1; total += r
            if term or trunc:
                break
        surv.append(steps); rew.append(total)
    env.close()
    return np.asarray(surv), np.asarray(rew)


def main():
    np.random.seed(0)
    w = Wasm(WASM_PATH)
    a_s, a_r = run(chaos)
    b_s, b_r = run(lambda o: w.action(o))
    lat = np.asarray(w.lat)
    print("\n" + "=" * 64)
    print(f" BIPEDALWALKER-v3 BENCHMARK  ({EPISODES} episodes)")
    print("=" * 64)
    print(f" A  CHAOS-ONLY  | survival {a_s.mean():6.1f} steps | reward {a_r.mean():8.2f}")
    print(f" B  WASM        | survival {b_s.mean():6.1f} steps | reward {b_r.mean():8.2f}")
    print("-" * 64)
    print(f" survival gain  : {(b_s.mean()/max(1,a_s.mean())-1)*100:+.0f}%   "
          f"reward delta {b_r.mean()-a_r.mean():+.1f}")
    print(f" wasm latency   : mean {lat.mean()*1000:.2f} us | max {lat.max()*1000:.2f} us")
    print("=" * 64)


if __name__ == "__main__":
    main()
