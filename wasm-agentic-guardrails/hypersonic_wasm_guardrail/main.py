"""
Hypersonic Kinetic Interceptor -- 6-DOF MuJoCo defense scenario with a WASM
structural-load safety envelope.

Pure native mujoco / mujoco.viewer. No external renderers, UI wrappers, or XML
assets -- the MJCF is generated programmatically below.

Scenario
--------
A single thrust-vectored interceptor defends against a multi-wave threat:
  Wave 1 (MIRV) : a cluster of 3-5 ballistic re-entry bodies on divergent
                  descent trajectories (a MIRV bus deployment).
  Wave 2 (HGV)  : 1-2 hypersonic glide vehicles that apply randomized lateral
                  forces to themselves while descending (evasive manoeuvring).

Flight model (sim-proxy units, rigid-body kinematics only -- no aero/weapons model)
  * Constant nose-axis THRUST -> the vehicle accelerates where it points;
    quadratic aero drag bounds cruise speed.
  * Pitch/yaw torque actuators slew the nose; aero angular damping prevents tumble.

Chaos Agent
-----------
Tracks the nearest threat and commands near-maximum torque to slew onto it as
fast as possible, ignoring structural limits.

WASM Safety Envelope
--------------------
At high speed, max deflection => excessive G-loading => structural failure.
guardrail.wasm evaluates Load = speed^2 * command against a hardcoded G-limit and
clamps to the maximum structurally safe deflection.
  envelope=True  : only the clamped command reaches the actuators (airframe safe).
  envelope=False : raw agent command is applied; the first time commanded load
                   exceeds the G-limit the airframe fails and the run aborts.
"""

import struct
import time
import random

import numpy as np
import mujoco

try:
    import mujoco.viewer as mj_viewer
except Exception:  # pragma: no cover
    mj_viewer = None

WASM_PATH = "guardrail.wasm"

# ---- pool of pre-allocated targets (MuJoCo bodies are static after compile) ----
N_MIRV_SLOTS = 5            # indices 0..4
N_HGV_SLOTS = 2             # indices 5..6
N_TARGETS = N_MIRV_SLOTS + N_HGV_SLOTS

# ---- scenario / flight tuning (sim-proxy units; tuned by live-fire sweep) ----
INIT_SPEED = 40.0           # interceptor boost speed at launch
THRUST = 2200.0             # constant nose-axis thrust
LIN_DRAG = 0.08             # quadratic aero drag -> terminal cruise ~166
ANG_DAMP = 90.0             # aerodynamic angular damping (prevents tumbling)
GEAR = 280.0                # actuator torque scale
GAIN = 6.0                  # agent guidance gain (high => near-max deflection)
G_LIMIT_DEFAULT = 6000.0    # structural load ceiling (proxy units)

SPAWN_ALT = 160.0           # target spawn altitude
DESCENT_SPEED = 40.0        # target descent rate
HIT_RADIUS = 24.0           # proximity-fuze lethal radius
LEAK_ALT = -60.0            # below this a target has leaked past the defender
HGV_LATERAL_FORCE = 1500.0  # random evasive force magnitude
WAVE2_TIME = 5.0            # seconds before the HGV wave releases
PARK = -1000.0              # parking altitude for inactive targets

OBS_N = 6
ACT_N = 2


# ----------------------------------------------------------------------------
# Programmatic MJCF (no external XML files)
# ----------------------------------------------------------------------------
def generate_mjcf(n_targets: int) -> str:
    targets = []
    for i in range(n_targets):
        targets.append(f"""
        <body name="target{i}" pos="0 0 {PARK - i * 5.0}">
          <freejoint name="tjoint{i}"/>
          <geom name="tgeom{i}" type="sphere" size="1.6" mass="20"
                contype="0" conaffinity="0" rgba="1.0 0.35 0.12 1"/>
        </body>""")
    target_xml = "".join(targets)

    return f"""
<mujoco model="hypersonic_intercept">
  <option timestep="0.004" gravity="0 0 -3.0" density="0.0005" viscosity="0.0"/>
  <visual>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.3 0.3 0.3"/>
    <map znear="0.01" zfar="2000"/>
  </visual>
  <worldbody>
    <geom name="ground" type="plane" size="0 0 1" pos="0 0 -45"
          contype="0" conaffinity="0" rgba="0.08 0.09 0.13 1"/>
    <body name="interceptor" pos="0 0 0">
      <freejoint name="ijoint"/>
      <geom name="igeom" type="cylinder" size="0.5 2.0" mass="60"
            contype="0" conaffinity="0" rgba="0.2 0.85 1.0 1"/>
      <site name="ctrl" pos="0 0 0" size="0.15" rgba="1 1 0 0.4"/>
    </body>{target_xml}
  </worldbody>
  <actuator>
    <motor name="pitch" site="ctrl" gear="0 0 0 0 {GEAR} 0" ctrlrange="-1 1"/>
    <motor name="yaw"   site="ctrl" gear="0 0 0 {GEAR} 0 0" ctrlrange="-1 1"/>
  </actuator>
</mujoco>"""


# ----------------------------------------------------------------------------
# WASM bridge
# ----------------------------------------------------------------------------
class Envelope:
    def __init__(self, path):
        from wasmtime import Engine, Store, Module, Instance
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
        self._set_g = ex["set_g_limit"]

    def set_g_limit(self, v):
        self._set_g(self.store, float(v))

    def filter(self, obs, cmd):
        self.mem.write(self.store, struct.pack(f"<{OBS_N}f", *obs), self.obs_off)
        self.mem.write(self.store, struct.pack(f"<{ACT_N}f", *cmd), self.act_in_off)
        t0 = time.perf_counter_ns()
        self._calc(self.store)
        lat_ms = (time.perf_counter_ns() - t0) / 1e6
        raw_a = self.mem.read(self.store, self.act_off, self.act_off + 4 * ACT_N)
        raw_s = self.mem.read(self.store, self.status_off, self.status_off + 4 * ACT_N)
        action = np.asarray(struct.unpack(f"<{ACT_N}f", raw_a), dtype=np.float32)
        status = struct.unpack(f"<{ACT_N}f", raw_s)
        return action, status, lat_ms


# ----------------------------------------------------------------------------
# Target bookkeeping
# ----------------------------------------------------------------------------
class Target:
    __slots__ = ("idx", "bid", "qadr", "vadr", "active", "kind")

    def __init__(self, idx, bid, qadr, vadr):
        self.idx = idx
        self.bid = bid
        self.qadr = qadr
        self.vadr = vadr
        self.active = False
        self.kind = None  # "MIRV" or "HGV"


def joint_addr(model, name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])


def spawn(data, tgt, pos, vel, kind):
    data.qpos[tgt.qadr:tgt.qadr + 3] = pos
    data.qpos[tgt.qadr + 3:tgt.qadr + 7] = [1, 0, 0, 0]
    data.qvel[tgt.vadr:tgt.vadr + 6] = [vel[0], vel[1], vel[2], 0, 0, 0]
    data.xfrc_applied[tgt.bid] = 0.0
    tgt.active = True
    tgt.kind = kind


def park(data, tgt):
    data.qpos[tgt.qadr:tgt.qadr + 3] = [0, 0, PARK - tgt.idx * 5.0]
    data.qpos[tgt.qadr + 3:tgt.qadr + 7] = [1, 0, 0, 0]
    data.qvel[tgt.vadr:tgt.vadr + 6] = 0.0
    data.xfrc_applied[tgt.bid] = 0.0
    tgt.active = False
    tgt.kind = None


# ----------------------------------------------------------------------------
# Chaos agent: reckless high-gain pursuit of the nearest threat
# ----------------------------------------------------------------------------
def chaos_command(rel_body):
    """Near-max deflection toward the target in body frame (z = nose axis)."""
    rx, ry, rz = rel_body
    pitch_err = np.arctan2(rx, rz)    # torque about +body-y swings nose toward +x
    yaw_err = np.arctan2(ry, rz)      # torque about +body-x swings nose toward -y
    pitch = float(np.clip(GAIN * pitch_err, -1.0, 1.0))
    yaw = float(np.clip(-GAIN * yaw_err, -1.0, 1.0))
    return np.array([pitch, yaw], dtype=np.float32)


def run(use_viewer=True, max_steps=9000, seed=0, verbose=True,
        g_limit=G_LIMIT_DEFAULT, envelope=True):
    random.seed(seed)
    np.random.seed(seed)

    model = mujoco.MjModel.from_xml_string(generate_mjcf(N_TARGETS))
    data = mujoco.MjData(model)
    env = Envelope(WASM_PATH)
    env.set_g_limit(g_limit)

    i_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "interceptor")
    i_qadr, i_vadr = joint_addr(model, "ijoint")

    targets = []
    for i in range(N_TARGETS):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"target{i}")
        qadr, vadr = joint_addr(model, f"tjoint{i}")
        targets.append(Target(i, bid, qadr, vadr))

    mujoco.mj_resetData(model, data)
    data.qvel[i_vadr:i_vadr + 3] = [0, 0, INIT_SPEED]
    for t in targets:
        park(data, t)

    n_mirv = random.randint(3, 5)
    for k in range(n_mirv):
        ang = 2 * np.pi * k / n_mirv
        pos = [12 * np.cos(ang), 12 * np.sin(ang), SPAWN_ALT]
        vel = [5 * np.cos(ang), 5 * np.sin(ang), -DESCENT_SPEED]
        spawn(data, targets[k], pos, vel, "MIRV")
    mujoco.mj_forward(model, data)

    stats = dict(intercepts=0, leakers=0, clamp_ticks=0, prevented=0, failures=0,
                 max_load=0.0, steps=0, lat=[], min_miss=1e18,
                 n_mirv=n_mirv, n_hgv=0, dead=False)
    state = {"wave2": False, "step": 0}

    def policy_and_step():
        i_pos = data.xpos[i_bid].copy()
        i_rot = data.xmat[i_bid].reshape(3, 3)
        lin = data.qvel[i_vadr:i_vadr + 3].copy()
        ang_body = data.qvel[i_vadr + 3:i_vadr + 6].copy()
        speed = float(np.linalg.norm(lin))

        nearest, best = None, 1e18
        for t in targets:
            if not t.active:
                continue
            d = float(np.linalg.norm(data.xpos[t.bid] - i_pos))
            if d < best:
                best, nearest = d, t
        if nearest is not None:
            stats["min_miss"] = min(stats["min_miss"], best)
            rel_body = i_rot.T @ (data.xpos[nearest.bid] - i_pos)
            rb = rel_body / (np.linalg.norm(rel_body) + 1e-6)
        else:
            rb = np.zeros(3)

        obs = [speed, float(ang_body[1]), float(ang_body[0]),
               float(rb[0]), float(rb[1]), float(rb[2])]
        chaos = chaos_command(rb)
        commanded_load = speed * speed * float(max(abs(chaos[0]), abs(chaos[1])))
        stats["max_load"] = max(stats["max_load"], commanded_load)

        if envelope:
            cmd, status, lat = env.filter(obs, chaos)
            stats["lat"].append(lat)
            if status[0] > 0.5 or status[1] > 0.5:
                stats["clamp_ticks"] += 1
                if commanded_load > g_limit:
                    stats["prevented"] += 1
            clamp_flag = "CLAMP" if (status[0] > 0.5 or status[1] > 0.5) else "pass "
        else:
            cmd = chaos
            clamp_flag = "RAW  "
            if commanded_load > g_limit and not stats["dead"]:
                stats["failures"] += 1
                stats["dead"] = True  # airframe lost

        if not stats["dead"]:
            data.ctrl[0] = float(cmd[0])
            data.ctrl[1] = float(cmd[1])
            nose_world = i_rot[:, 2]
            world_ang = i_rot @ ang_body
            data.xfrc_applied[i_bid, 0:3] = nose_world * THRUST - LIN_DRAG * speed * lin
            data.xfrc_applied[i_bid, 3:6] = -ANG_DAMP * world_ang
        else:
            data.ctrl[0] = 0.0
            data.ctrl[1] = 0.0
            data.xfrc_applied[i_bid] = 0.0

        for t in targets:
            if t.active and t.kind == "HGV":
                data.xfrc_applied[t.bid, 0:3] = [
                    np.random.uniform(-HGV_LATERAL_FORCE, HGV_LATERAL_FORCE),
                    np.random.uniform(-HGV_LATERAL_FORCE, HGV_LATERAL_FORCE),
                    0.0,
                ]

        mujoco.mj_step(model, data)
        state["step"] += 1
        stats["steps"] = state["step"]

        i_pos2 = data.xpos[i_bid]
        for t in targets:
            if not t.active:
                continue
            tp = data.xpos[t.bid]
            if (not stats["dead"]) and np.linalg.norm(tp - i_pos2) < HIT_RADIUS:
                stats["intercepts"] += 1
                park(data, t)
            elif tp[2] < LEAK_ALT:
                stats["leakers"] += 1
                park(data, t)

        if (not state["wave2"]) and data.time >= WAVE2_TIME:
            n_hgv = random.randint(1, 2)
            stats["n_hgv"] = n_hgv
            for j in range(n_hgv):
                slot = N_MIRV_SLOTS + j
                pos = [random.uniform(-20, 20), random.uniform(-20, 20), SPAWN_ALT]
                spawn(data, targets[slot], pos, [0, 0, -DESCENT_SPEED * 0.85], "HGV")
            state["wave2"] = True

        if verbose and state["step"] % 100 == 0:
            active = sum(1 for t in targets if t.active)
            print(f"[t={data.time:6.2f}] spd={speed:6.1f} miss={best:7.1f} "
                  f"load={commanded_load:9.1f} {clamp_flag} | active={active} "
                  f"hits={stats['intercepts']} leak={stats['leakers']} "
                  f"{'DEAD' if stats['dead'] else ''}")

    def engagement_over():
        if stats["dead"] and not any(t.active for t in targets):
            return True
        return state["wave2"] and not any(t.active for t in targets)

    # ------------------------------------------------------------------ viewer
    if use_viewer and mj_viewer is not None:
        try:
            with mj_viewer.launch_passive(model, data) as viewer:
                # Camera tracks the interceptor body so it never leaves frame.
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = i_bid
                viewer.cam.distance = 120.0
                viewer.cam.azimuth = 90.0
                viewer.cam.elevation = -20.0

                dt = model.opt.timestep
                summarized = False

                while viewer.is_running():
                    tic = time.perf_counter()

                    if (not engagement_over()) and state["step"] < max_steps:
                        # Active engagement: advance the physics one timestep.
                        policy_and_step()
                    elif not summarized:
                        # Engagement resolved: freeze physics, report, keep window.
                        _summary(stats, g_limit, envelope)
                        print(">>> Engagement resolved -- physics frozen on final "
                              "frame. Close the viewer window to exit.")
                        summarized = True

                    # Keep rendering whether stepping or frozen.
                    viewer.sync()

                    # Real-time pacing: hold each iteration to one physics timestep.
                    remaining = dt - (time.perf_counter() - tic)
                    if remaining > 0:
                        time.sleep(remaining)

            if not summarized:
                _summary(stats, g_limit, envelope)
            return stats
        except Exception as exc:
            print(f"[viewer unavailable: {exc}] -- running headless")

    # ----------------------------------------------------------------- headless
    while state["step"] < max_steps and not engagement_over():
        policy_and_step()
    _summary(stats, g_limit, envelope)
    return stats


def _summary(stats, g_limit, envelope):
    lat = np.asarray(stats["lat"]) if stats["lat"] else np.zeros(1)
    total = stats["n_mirv"] + stats["n_hgv"]
    mode = "WASM ENVELOPE" if envelope else "NO ENVELOPE (raw chaos)"
    print("\n" + "=" * 66)
    print(f" HYPERSONIC INTERCEPT -- {mode}")
    print("=" * 66)
    print(f" threats (MIRV+HGV)         : {stats['n_mirv']} + {stats['n_hgv']} = {total}")
    print(f" targets destroyed          : {stats['intercepts']} / {total}")
    print(f" leakers (escaped)          : {stats['leakers']}")
    print(f" peak commanded load        : {stats['max_load']:.0f}  (G-limit {g_limit:.0f})")
    if envelope:
        print(f" ticks clamped              : {stats['clamp_ticks']} / {stats['steps']}")
        print(f" structural failures        : 0  (envelope held load <= limit)")
        print(f" failures prevented (clamps): {stats['prevented']}")
        print(f" wasm latency               : mean {lat.mean()*1000:.2f} us | "
              f"max {lat.max()*1000:.2f} us")
    else:
        print(f" structural failures        : {stats['failures']}  (airframe lost)")
    print("=" * 66)


if __name__ == "__main__":
    run(use_viewer=True)
