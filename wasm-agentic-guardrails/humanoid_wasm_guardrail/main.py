"""
Humanoid-v4 WASM collapse-prevention envelope.

The Chaos Agent drives all 17 actuators to their bounds, instantly hyperextending
the joints and collapsing the centre of mass into the floor. guardrail.wasm watches
the torso z-height: when a fall is imminent it clamps the lower-body / core
actuators (abdomen + hips + knees) to a rigid stabilizing posture, leaving the arms
under agent control. Only the filtered action reaches env.step().

Usage:
    python main.py              # interactive render window (Chaos through envelope)
    python main.py --benchmark  # headless A/B: chaos-only vs chaos-through-envelope
"""

import sys
import struct
import time

import numpy as np
import gymnasium as gym
from wasmtime import Engine, Store, Module, Instance

WASM_PATH = "guardrail.wasm"
OBS_WRITE = 45     # telemetry handed to the wasm: OBS[0]=z, [5:22]=joint angles, [28:45]=joint vels
ACT_N = 17


def make_env(render=False):
    """Humanoid-v4, falling back to v5 if the registry has moved on."""
    kwargs = {"render_mode": "human"} if render else {}
    for env_id in ("Humanoid-v4", "Humanoid-v5"):
        try:
            return gym.make(env_id, **kwargs)
        except Exception:
            continue
    raise RuntimeError("No Humanoid env available (install gymnasium[mujoco]).")


class Envelope:
    def __init__(self, path):
        engine = Engine()
        self.store = Store(engine)
        module = Module.from_file(engine, path)
        inst = Instance(self.store, module, [])
        ex = inst.exports(self.store)
        self.mem = ex["memory"]
        self.obs_off = ex["obs_ptr"](self.store)
        self.act_in_off = ex["act_in_ptr"](self.store)
        self.act_off = ex["act_ptr"](self.store)
        self.status_off = ex["status_ptr"](self.store)
        self._calc = ex["calculate_correction"]
        self._set_z = ex["set_z_critical"]
        self._set_gains = ex["set_gains"]

    def set_z_critical(self, v):
        self._set_z(self.store, float(v))

    def set_gains(self, kp, kd):
        self._set_gains(self.store, float(kp), float(kd))

    def filter(self, obs, act_in):
        self.mem.write(self.store, struct.pack(f"<{OBS_WRITE}f", *obs[:OBS_WRITE]), self.obs_off)
        self.mem.write(self.store, struct.pack(f"<{ACT_N}f", *act_in), self.act_in_off)
        t0 = time.perf_counter_ns()
        self._calc(self.store)
        lat_ms = (time.perf_counter_ns() - t0) / 1e6
        raw_a = self.mem.read(self.store, self.act_off, self.act_off + 4 * ACT_N)
        raw_s = self.mem.read(self.store, self.status_off, self.status_off + 4)
        action = np.asarray(struct.unpack(f"<{ACT_N}f", raw_a), dtype=np.float32)
        status = struct.unpack("<f", raw_s)[0]
        return action, status, lat_ms


def chaos_agent(action_space):
    """Maximum-bound erratic action: every channel pinned to a random extreme."""
    pick = np.random.rand(ACT_N) > 0.5
    return np.where(pick, action_space.high, action_space.low).astype(np.float32)


# ----------------------------------------------------------------------------
# Interactive run (render window)
# ----------------------------------------------------------------------------
def interactive():
    env = make_env(render=True)
    envelope = Envelope(WASM_PATH)
    obs, _ = env.reset()
    tick = 0
    interventions = 0
    while True:
        chaos = chaos_agent(env.action_space)
        action, status, lat = envelope.filter(obs, chaos)
        if status > 0.5:
            interventions += 1
        print(f"[t={tick:05d}] z={obs[0]:+.3f} | "
              f"{'STABILIZE' if status > 0.5 else 'pass-thru'} | "
              f"interventions={interventions} | wasm_lat={lat:.4f} ms "
              f"{'OK' if lat < 0.1 else 'SLOW'}")
        obs, _, terminated, truncated, _ = env.step(action)
        tick += 1
        if terminated or truncated:
            print(f"\n>>> Episode ended at t={tick}. Resetting. <<<\n")
            obs, _ = env.reset()
            tick = 0
            interventions = 0


# ----------------------------------------------------------------------------
# Headless A/B benchmark
# ----------------------------------------------------------------------------
def benchmark(episodes=20, max_steps=1000):
    env = make_env(render=False)
    envelope = Envelope(WASM_PATH)
    envelope.set_z_critical(1.38)   # engage stance-hold the instant height dips
    envelope.set_gains(16.0, 0.5)   # tuned by survival sweep

    def run_condition(use_envelope):
        survival, survived_cap, interv_ticks, total_ticks = [], 0, 0, 0
        lat = []
        for ep in range(episodes):
            obs, _ = env.reset(seed=1000 + ep)
            steps = 0
            while steps < max_steps:
                chaos = chaos_agent(env.action_space)
                if use_envelope:
                    action, status, l = envelope.filter(obs, chaos)
                    lat.append(l)
                    if status > 0.5:
                        interv_ticks += 1
                else:
                    action = chaos
                total_ticks += 1
                obs, _, terminated, truncated, _ = env.step(action)
                steps += 1
                if terminated or truncated:
                    break
            survival.append(steps)
            if steps >= max_steps:
                survived_cap += 1
        return dict(mean=np.mean(survival), median=np.median(survival),
                    survived_cap=survived_cap, interv=interv_ticks,
                    total=total_ticks, lat=lat)

    a = run_condition(False)
    b = run_condition(True)
    env.close()

    print("\n" + "=" * 78)
    print(f" HUMANOID-v4 COLLAPSE-PREVENTION BENCHMARK  ({episodes} episodes, cap {max_steps})")
    print("=" * 78)
    print(f" {'condition':<26} | {'mean steps':>10} | {'median':>7} | {'survived cap':>12}")
    print(" " + "-" * 74)
    print(f" {'A  CHAOS-ONLY':<26} | {a['mean']:>10.1f} | {a['median']:>7.0f} | "
          f"{a['survived_cap']:>10d}/{episodes}")
    print(f" {'B  CHAOS->ENVELOPE':<26} | {b['mean']:>10.1f} | {b['median']:>7.0f} | "
          f"{b['survived_cap']:>10d}/{episodes}")
    print(" " + "-" * 74)
    lat = np.asarray(b["lat"])
    interv_rate = 100.0 * b["interv"] / max(1, b["total"])
    gain = (b["mean"] / a["mean"] - 1.0) * 100.0 if a["mean"] > 0 else 0.0
    print(f" survival gain (B vs A) : {gain:+.1f}%   "
          f"(mean {a['mean']:.1f} -> {b['mean']:.1f} steps)")
    print(f" envelope intervention  : {interv_rate:.1f}% of ticks engaged stabilize")
    print(f" wasm latency           : mean {lat.mean()*1000:.2f} us | "
          f"p99 {np.percentile(lat,99)*1000:.2f} us | max {lat.max()*1000:.2f} us")
    print("=" * 78)


if __name__ == "__main__":
    if "--benchmark" in sys.argv:
        benchmark()
    else:
        interactive()
