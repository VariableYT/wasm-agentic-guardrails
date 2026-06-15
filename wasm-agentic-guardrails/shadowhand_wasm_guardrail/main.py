"""
Shadow Hand (HandManipulateBlock-v1) WASM joint-limit safety envelope.

The Chaos Agent issues violent random commands across the 20 finger actuators,
slamming joints into their mechanical hard-stops (hyperextension / finger
intersection -- servo death on real hardware). guardrail.wasm works in normalised
joint space: it clamps every command into a safe operating band and predictively
halts any servo about to cross its limit. Only the safe command reaches env.step().

Usage:
    python main.py              # interactive render window
    python main.py --benchmark  # headless A/B: chaos-only vs chaos-through-envelope
"""

import sys
import struct
import time

import numpy as np
import gymnasium as gym
import gymnasium_robotics
import mujoco

gym.register_envs(gymnasium_robotics)

WASM_PATH = "guardrail.wasm"
N_ACT = 20
OBS_WRITE = 40           # 20 normalised positions + 20 normalised velocities
HARDSTOP = 0.98          # |normalised pos| above this = mechanical-limit violation


def make_env(render=False):
    kwargs = {"render_mode": "human"} if render else {}
    return gym.make("HandManipulateBlock-v1", **kwargs)


def build_joint_map(model):
    """For each of the 20 actuators: (qpos addr, qvel addr, lo, hi)."""
    qadr, dofadr, lo, hi = [], [], [], []
    for i in range(model.nu):
        jid = int(model.actuator_trnid[i, 0])
        qadr.append(int(model.jnt_qposadr[jid]))
        dofadr.append(int(model.jnt_dofadr[jid]))
        lo.append(float(model.jnt_range[jid, 0]))
        hi.append(float(model.jnt_range[jid, 1]))
    return (np.array(qadr), np.array(dofadr),
            np.array(lo, dtype=np.float32), np.array(hi, dtype=np.float32))


class JointTelemetry:
    """Reads live joint state and normalises it so +/-1.0 == mechanical limit."""

    def __init__(self, model):
        self.qadr, self.dofadr, self.lo, self.hi = build_joint_map(model)
        self.span = (self.hi - self.lo)

    def normalized(self, data):
        pos = np.asarray(data.qpos[self.qadr], dtype=np.float32)
        vel = np.asarray(data.qvel[self.dofadr], dtype=np.float32)
        pos_n = 2.0 * (pos - self.lo) / self.span - 1.0
        vel_n = vel * (2.0 / self.span)
        return pos_n, vel_n


class Envelope:
    def __init__(self, path):
        from wasmtime import Engine, Store, Module, Instance
        engine = Engine()
        self.store = Store(engine)
        module = Module.from_file(engine, path)
        ex = Instance(self.store, module, []).exports(self.store)
        self.mem = ex["memory"]
        self.obs_off = ex["obs_ptr"](self.store)
        self.act_in_off = ex["act_in_ptr"](self.store)
        self.act_off = ex["act_ptr"](self.store)
        self.status_off = ex["status_ptr"](self.store)
        self._calc = ex["calculate_correction"]
        self._set_safe = ex["set_safe_limit"]
        self._set_look = ex["set_lookahead"]

    def set_safe_limit(self, v):
        self._set_safe(self.store, float(v))

    def set_lookahead(self, v):
        self._set_look(self.store, float(v))

    def filter(self, pos_n, vel_n, act_in):
        obs = np.concatenate([pos_n, vel_n]).astype(np.float32)
        self.mem.write(self.store, struct.pack(f"<{OBS_WRITE}f", *obs), self.obs_off)
        self.mem.write(self.store, struct.pack(f"<{N_ACT}f", *act_in), self.act_in_off)
        t0 = time.perf_counter_ns()
        self._calc(self.store)
        lat_ms = (time.perf_counter_ns() - t0) / 1e6
        raw_a = self.mem.read(self.store, self.act_off, self.act_off + 4 * N_ACT)
        raw_s = self.mem.read(self.store, self.status_off, self.status_off + 4)
        action = np.asarray(struct.unpack(f"<{N_ACT}f", raw_a), dtype=np.float32)
        clamped = struct.unpack("<f", raw_s)[0]
        return action, clamped, lat_ms


def chaos_agent():
    """Violent, maximum-bound random command on every finger channel."""
    return np.where(np.random.rand(N_ACT) > 0.5, 1.0, -1.0).astype(np.float32)


# ----------------------------------------------------------------------------
# Interactive run (render window)
# ----------------------------------------------------------------------------
def interactive():
    env = make_env(render=True)
    tel = JointTelemetry(env.unwrapped.model)
    envelope = Envelope(WASM_PATH)
    env.reset()
    data = env.unwrapped.data
    tick = 0
    prevented = 0
    while True:
        pos_n, vel_n = tel.normalized(data)
        chaos = chaos_agent()
        action, clamped, lat = envelope.filter(pos_n, vel_n, chaos)
        prevented += int(clamped)
        hardstops = int(np.sum(np.abs(pos_n) > HARDSTOP))
        print(f"[t={tick:05d}] clamped={int(clamped):2d}/20 | hardstop_joints={hardstops:2d} "
              f"| prevented_total={prevented} | wasm_lat={lat:.4f} ms "
              f"{'OK' if lat < 0.1 else 'SLOW'}")
        _, _, terminated, truncated, _ = env.step(action)
        tick += 1
        if terminated or truncated:
            env.reset()
            tick = 0


# ----------------------------------------------------------------------------
# Headless A/B benchmark
# ----------------------------------------------------------------------------
def benchmark(episodes=20):
    env = make_env(render=False)
    tel = JointTelemetry(env.unwrapped.model)
    envelope = Envelope(WASM_PATH)
    data = env.unwrapped.data

    def run_condition(use_envelope):
        hardstop_ticks = 0     # joint-steps spent past the mechanical limit
        prevented = 0          # channels the envelope clamped
        total_ch = 0
        steps = 0
        lat = []
        for ep in range(episodes):
            env.reset(seed=2000 + ep)
            done = False
            while not done:
                pos_n, vel_n = tel.normalized(data)
                hardstop_ticks += int(np.sum(np.abs(pos_n) > HARDSTOP))
                chaos = chaos_agent()
                if use_envelope:
                    action, clamped, l = envelope.filter(pos_n, vel_n, chaos)
                    lat.append(l)
                    prevented += int(clamped)
                    total_ch += N_ACT
                else:
                    action = chaos
                _, _, terminated, truncated, _ = env.step(action)
                steps += 1
                done = terminated or truncated
        return dict(hardstop=hardstop_ticks, prevented=prevented,
                    total=total_ch, steps=steps, lat=lat)

    a = run_condition(False)
    b = run_condition(True)
    env.close()

    print("\n" + "=" * 80)
    print(f" SHADOW HAND JOINT-LIMIT BENCHMARK  ({episodes} episodes)")
    print("=" * 80)
    print(f" {'condition':<26} | {'steps':>6} | {'hard-stop joint-ticks':>22}")
    print(" " + "-" * 76)
    print(f" {'A  CHAOS-ONLY':<26} | {a['steps']:>6} | {a['hardstop']:>22}")
    print(f" {'B  CHAOS->ENVELOPE':<26} | {b['steps']:>6} | {b['hardstop']:>22}")
    print(" " + "-" * 76)
    reduction = (1 - b["hardstop"] / max(1, a["hardstop"])) * 100.0
    net_prevented = a["hardstop"] - b["hardstop"]
    interv = 100.0 * b["prevented"] / max(1, b["total"])
    lat = np.asarray(b["lat"])
    print(f" hyperextension reduction : {reduction:.1f}%  "
          f"({a['hardstop']} -> {b['hardstop']} joint-ticks past limit)")
    print(f" violations prevented     : {net_prevented} joint-ticks past limit "
          f"(A hard-stop count minus B)")
    print(f" channel-clamp activations: {b['prevented']} ({interv:.1f}% of commands) -- "
          f"inflated: the Chaos Agent always commands max deflection,")
    print(f"                            so every channel is shaved 1.0 -> safe band each tick")
    print(f" wasm latency             : mean {lat.mean()*1000:.2f} us | "
          f"p99 {np.percentile(lat,99)*1000:.2f} us | max {lat.max()*1000:.2f} us")
    print("=" * 80)


if __name__ == "__main__":
    if "--benchmark" in sys.argv:
        benchmark()
    else:
        interactive()
