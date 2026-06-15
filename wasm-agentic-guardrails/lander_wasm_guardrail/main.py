"""
LunarLanderContinuous WASM safety-envelope bridge (true intercept filter).

Every tick the Chaos Agent proposes an action. The host writes BOTH the live
observation AND the proposed action into guardrail.wasm, then calls
calculate_correction(). The WASM evaluates the proposal against its deterministic
PD safety envelope: actions inside the trust band pass through untouched; only
destructive deviations are clamped to the safe PD command. ONLY the filtered
action reaches env.step().
"""

import struct
import time

import numpy as np
import gymnasium as gym
from wasmtime import Engine, Store, Module, Instance

WASM_PATH = "guardrail.wasm"
OBS_N = 8
ACT_N = 2


def make_env():
    """v2 was retired in gymnasium>=1.0; fall back to v3 automatically."""
    for env_id in ("LunarLanderContinuous-v2", "LunarLanderContinuous-v3"):
        try:
            return gym.make(env_id, render_mode="human")
        except Exception:
            continue
    raise RuntimeError("No LunarLanderContinuous env available (install gymnasium[box2d]).")


def write_floats(store, mem, offset, values):
    mem.write(store, struct.pack(f"<{len(values)}f", *values), offset)


def read_floats(store, mem, offset, n):
    raw = mem.read(store, offset, offset + 4 * n)
    return list(struct.unpack(f"<{n}f", raw))


def chaos_agent():
    """Maximum-bound erratic thrust: each engine pinned to a random extreme."""
    return np.where(np.random.rand(2) > 0.5, 1.0, -1.0).astype(np.float32)


def main():
    engine = Engine()
    store = Store(engine)
    module = Module.from_file(engine, WASM_PATH)
    instance = Instance(store, module, [])
    exports = instance.exports(store)

    mem = exports["memory"]
    obs_off = exports["obs_ptr"](store)
    act_in_off = exports["act_in_ptr"](store)
    act_off = exports["act_ptr"](store)
    status_off = exports["status_ptr"](store)
    calculate_correction = exports["calculate_correction"]

    env = make_env()
    obs, _ = env.reset()

    tick = 0
    while True:
        chaos = chaos_agent()

        # Feed BOTH telemetry and the agent's proposed action into the envelope.
        write_floats(store, mem, obs_off, [float(v) for v in obs[:OBS_N]])
        write_floats(store, mem, act_in_off, [float(v) for v in chaos])

        t0 = time.perf_counter_ns()
        calculate_correction(store)
        t1 = time.perf_counter_ns()

        action = np.asarray(read_floats(store, mem, act_off, ACT_N), dtype=np.float32)
        status = read_floats(store, mem, status_off, ACT_N)  # per-channel 0/1

        latency_ms = (t1 - t0) / 1e6
        clamped = int(sum(status))
        verdict = ["PASS", "PASS"]
        if status[0] > 0.5:
            verdict[0] = "CLAMP"
        if status[1] > 0.5:
            verdict[1] = "CLAMP"
        print(
            f"[t={tick:05d}] chaos=[{chaos[0]:+.2f},{chaos[1]:+.2f}] -> "
            f"safe=[{action[0]:+.2f},{action[1]:+.2f}] "
            f"main:{verdict[0]} lat:{verdict[1]} | clamped={clamped}/2 | "
            f"y={obs[1]:+.3f} | wasm_lat={latency_ms:.5f} ms "
            f"{'OK' if latency_ms < 0.1 else 'SLOW'}"
        )

        # Capture the reward instead of ignoring it with '_'
        obs, reward, terminated, truncated, _ = env.step(action)

        # Accumulate the score (handle it gracefully if we just initialized)
        if 'episode_score' not in locals():
            episode_score = 0.0
            episode_count = 0

        episode_score += float(reward)
        tick += 1

        if terminated or truncated:
            episode_count += 1
            print(f"\n>>> [EPISODE {episode_count} SECURED] Final Score: {episode_score:.2f} <<<\n")
            obs, _ = env.reset()
            episode_score = 0.0  # Reset for the next drop


if __name__ == "__main__":
    main()
