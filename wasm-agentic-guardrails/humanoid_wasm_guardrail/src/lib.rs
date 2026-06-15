#![no_std]
//! Humanoid-v4 collapse-prevention safety envelope (joint-space PD stance hold).
//!
//! The hazard: the Chaos Agent slams all 17 actuators to their bounds, instantly
//! hyperextending hips/knees and dropping the centre of mass into the floor.
//!
//! The filter watches the torso Z-height (OBS[0]). When it falls below a critical
//! threshold (imminent fall), the lower-body / core actuators (abdomen + both hips
//! + both knees, action indices 0..=10) are driven by a deterministic PD law toward
//! an upright stance -- torque = Kp*(0 - angle) - Kd*angular_velocity -- instead of
//! passing the agent's destabilizing command. Arms (11..=16) stay under agent
//! control. Above the threshold every channel passes through.
//!
//! Linear-memory model (indices verified against Humanoid-v4, x/y excluded):
//!   OBS[45] : OBS[0]      = torso z-height
//!             OBS[5..22]  = 17 joint angles
//!             OBS[28..45] = 17 joint angular velocities
//!   ACT_IN[17] = agent's proposed action (bounded +/-0.4)            host -> wasm
//!   ACT[17]    = safe action emitted to the actuators                wasm -> host
//!   STATUS[1]  = 1.0 if the PD stance-hold engaged this tick          wasm -> host

use core::panic::PanicInfo;
use core::ptr::addr_of_mut;

#[panic_handler]
fn panic(_: &PanicInfo) -> ! {
    loop {}
}

const N_OBS: usize = 45;
const N_ACT: usize = 17;
const N_LEG: usize = 11; // abdomen(3) + hips(6) + knees(2)
const ACT_LIMIT: f32 = 0.4;
const ANG_OFF: usize = 5; // joint angles begin at OBS[5]
const VEL_OFF: usize = 28; // joint velocities begin at OBS[28]

static mut OBS: [f32; N_OBS] = [0.0; N_OBS];
static mut ACT_IN: [f32; N_ACT] = [0.0; N_ACT];
static mut ACT: [f32; N_ACT] = [0.0; N_ACT];
static mut STATUS: [f32; 1] = [0.0; 1];

/// Torso z-height below which the stance-hold engages. Runtime-tunable.
/// (Reset height is ~1.39; 1.38 engages the stabilizer the instant height dips.)
static mut Z_CRITICAL: f32 = 1.38;
/// PD gains toward the upright stance (runtime-tunable; tuned by survival sweep).
static mut KP: f32 = 16.0;
static mut KD: f32 = 0.5;

#[no_mangle]
pub extern "C" fn obs_ptr() -> *mut f32 {
    addr_of_mut!(OBS) as *mut f32
}

#[no_mangle]
pub extern "C" fn act_in_ptr() -> *mut f32 {
    addr_of_mut!(ACT_IN) as *mut f32
}

#[no_mangle]
pub extern "C" fn act_ptr() -> *mut f32 {
    addr_of_mut!(ACT) as *mut f32
}

#[no_mangle]
pub extern "C" fn status_ptr() -> *mut f32 {
    addr_of_mut!(STATUS) as *mut f32
}

#[no_mangle]
pub extern "C" fn set_z_critical(v: f32) {
    unsafe {
        *addr_of_mut!(Z_CRITICAL) = v;
    }
}

#[no_mangle]
pub extern "C" fn set_gains(kp: f32, kd: f32) {
    unsafe {
        *addr_of_mut!(KP) = kp;
        *addr_of_mut!(KD) = kd;
    }
}

#[inline(always)]
fn clamp(x: f32, lo: f32, hi: f32) -> f32 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

/// Evaluate the agent's action against the collapse envelope.
#[no_mangle]
pub extern "C" fn calculate_correction() {
    unsafe {
        let o = &*addr_of_mut!(OBS);
        let z = o[0];
        let collapsing = z < *addr_of_mut!(Z_CRITICAL);
        let kp = *addr_of_mut!(KP);
        let kd = *addr_of_mut!(KD);

        let inp = &*addr_of_mut!(ACT_IN);
        let a = &mut *addr_of_mut!(ACT);
        let s = &mut *addr_of_mut!(STATUS);

        let mut i = 0usize;
        while i < N_ACT {
            let cmd = clamp(inp[i], -ACT_LIMIT, ACT_LIMIT);
            if collapsing && i < N_LEG {
                // PD stance hold: restoring torque toward the upright posture (target 0).
                let angle = o[ANG_OFF + i];
                let vel = o[VEL_OFF + i];
                let torque = kp * (0.0 - angle) - kd * vel;
                a[i] = clamp(torque, -ACT_LIMIT, ACT_LIMIT);
            } else {
                a[i] = cmd;
            }
            i += 1;
        }
        s[0] = if collapsing { 1.0 } else { 0.0 };
    }
}
