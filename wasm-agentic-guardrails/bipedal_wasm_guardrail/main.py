"""
BipedalWalker-v3 WASM safety-envelope bridge.

The Chaos Agent drives all four motors to random extreme torque. guardrail.wasm
replaces it with a deterministic PD hull-stabilizer + bounded gait oscillator.
"""

import struct
import time

import numpy as np
import gymnasium as gym
from wasmtime import Engine, Store, Module, Instance

WASM_PATH = "guardrail.wasm"
OBS_N = 24
ACT_N = 4


def write_floats(store, mem, offset, values):
    mem.write(store, struct.pack(f"<{len(values)}f", *values), offset)


def read_floats(store, mem, offset, n):
    raw = mem.read(store, offset, offset + 4 * n)
    return list(struct.unpack(f"<{n}f", raw))


def chaos_agent():
    """Maximum-bound erratic torque: every motor pinned to a random extreme."""
    return np.where(np.random.rand(4) > 0.5, 1.0, -1.0).astype(np.float32)


def main():
    engine = Engine()
    store = Store(engine)
    module = Module.from_file(engine, WASM_PATH)
    instance = Instance(store, module, [])
    exports = instance.exports(store)

    mem = exports["memory"]
    obs_off = exports["obs_ptr"](store)
    act_off = exports["act_ptr"](store)
    calculate_correction = exports["calculate_correction"]

    env = gym.make("BipedalWalker-v3", render_mode="human")
    obs, _ = env.reset()

    tick = 0
    while True:
        chaos = chaos_agent()

        write_floats(store, mem, obs_off, [float(v) for v in obs[:OBS_N]])
        t0 = time.perf_counter_ns()
        calculate_correction(store)
        t1 = time.perf_counter_ns()
        action = np.asarray(read_floats(store, mem, act_off, ACT_N), dtype=np.float32)

        latency_ms = (t1 - t0) / 1e6
        delta = float(np.abs(action - chaos).sum())
        print(
            f"[t={tick:05d}] chaos_sum={float(chaos.sum()):+.1f} -> "
            f"wasm=[{action[0]:+.2f},{action[1]:+.2f},{action[2]:+.2f},{action[3]:+.2f}] "
            f"| delta={delta:.2f} | hull={obs[0]:+.3f} | wasm_lat={latency_ms:.5f} ms "
            f"{'OK' if latency_ms < 0.1 else 'SLOW'}"
        )

        obs, _, terminated, truncated, _ = env.step(action)
        tick += 1
        if terminated or truncated:
            obs, _ = env.reset()


if __name__ == "__main__":
    main()
