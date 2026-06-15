"""
CartPole-v1 WASM safety-envelope bridge.

The Chaos Agent emits maximum-bound random actions (a 'hallucinating' policy).
Every tick the action is intercepted: the live observation is pushed into the
guardrail.wasm shared memory, the deterministic correction is computed in WASM,
and ONLY the corrected action reaches env.step().
"""

import struct
import time

import numpy as np
import gymnasium as gym
from wasmtime import Engine, Store, Module, Instance

WASM_PATH = "guardrail.wasm"
OBS_N = 4
ACT_N = 1


def write_floats(store, mem, offset, values):
    mem.write(store, struct.pack(f"<{len(values)}f", *values), offset)


def read_floats(store, mem, offset, n):
    raw = mem.read(store, offset, offset + 4 * n)
    return list(struct.unpack(f"<{n}f", raw))


def chaos_agent():
    """Erratic, untrained 'hallucinating' policy: uniformly random discrete action."""
    return int(np.random.randint(0, 2))


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

    env = gym.make("CartPole-v1", render_mode="human")
    obs, _ = env.reset()

    tick = 0
    while True:
        chaos = chaos_agent()

        write_floats(store, mem, obs_off, [float(v) for v in obs[:OBS_N]])
        t0 = time.perf_counter_ns()
        calculate_correction(store)
        t1 = time.perf_counter_ns()
        safe = read_floats(store, mem, act_off, ACT_N)
        action = int(round(safe[0]))

        latency_ms = (t1 - t0) / 1e6
        override = "OVERRIDE" if action != chaos else "passthrough"
        print(
            f"[t={tick:05d}] chaos={chaos} -> wasm={action} "
            f"({override}) | theta={obs[2]:+.4f} | wasm_lat={latency_ms:.5f} ms "
            f"{'OK' if latency_ms < 0.1 else 'SLOW'}"
        )

        obs, _, terminated, truncated, _ = env.step(action)
        tick += 1
        if terminated or truncated:
            obs, _ = env.reset()


if __name__ == "__main__":
    main()
