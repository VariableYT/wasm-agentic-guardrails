"""
Headless A/B benchmark for the CartPole WASM guardrail.
  A  CHAOS-ONLY : random left/right actions.
  B  WASM       : deterministic full-state-feedback controller in guardrail.wasm.
Metric: pole-balance survival (steps before the pole falls / episode truncates).
"""
import struct, time
import numpy as np
import gymnasium as gym
from wasmtime import Engine, Store, Module, Instance

WASM_PATH = "guardrail.wasm"
OBS_N, ACT_N = 4, 1
EPISODES = 50


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
        return int(round(struct.unpack(f"<{ACT_N}f", raw)[0]))


def run(policy):
    env = gym.make("CartPole-v1")
    surv = []
    for ep in range(EPISODES):
        obs, _ = env.reset(seed=3000 + ep); steps = 0
        while True:
            obs, _, term, trunc, _ = env.step(policy(obs, env)); steps += 1
            if term or trunc:
                break
        surv.append(steps)
    env.close()
    return np.asarray(surv)


def main():
    w = Wasm(WASM_PATH)
    a = run(lambda o, e: e.action_space.sample())
    b = run(lambda o, e: w.action(o))
    lat = np.asarray(w.lat)
    print("\n" + "=" * 60)
    print(f" CARTPOLE-v1 BENCHMARK  ({EPISODES} episodes, cap 500)")
    print("=" * 60)
    print(f" A  CHAOS-ONLY  | mean survival {a.mean():6.1f} | median {np.median(a):4.0f} steps")
    print(f" B  WASM        | mean survival {b.mean():6.1f} | median {np.median(b):4.0f} steps")
    print("-" * 60)
    print(f" balanced to cap (500): A {(a>=500).sum()}/{EPISODES}  ->  B {(b>=500).sum()}/{EPISODES}")
    print(f" survival gain  : {(b.mean()/a.mean()-1)*100:+.0f}%")
    print(f" wasm latency   : mean {lat.mean()*1000:.2f} us | max {lat.max()*1000:.2f} us")
    print("=" * 60)


if __name__ == "__main__":
    main()
