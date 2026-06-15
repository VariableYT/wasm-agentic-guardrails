#![no_std]
//! BipedalWalker-v3 deterministic safety envelope.
//! OBS[24] = [hull_angle, hull_ang_vel, vx, vy,
//!            hip1, hip1_speed, knee1, knee1_speed, leg1_contact,
//!            hip2, hip2_speed, knee2, knee2_speed, leg2_contact,
//!            lidar0..lidar9]
//! ACT[4] = [hip1_torque, knee1_torque, hip2_torque, knee2_torque]  each in -1..1
//!
//! This is an honest STABILIZER: a PD hull-leveler plus a bounded gait
//! oscillator (pure arithmetic, no libm/trig), all hard-clamped to action
//! bounds. It keeps the walker upright and inside the safety envelope.

use core::panic::PanicInfo;
use core::ptr::addr_of_mut;

#[panic_handler]
fn panic(_: &PanicInfo) -> ! {
    loop {}
}

static mut OBS: [f32; 24] = [0.0; 24];
static mut ACT: [f32; 4] = [0.0; 4];
static mut PHASE: f32 = 0.0; // gait phase in [0, 1)

#[no_mangle]
pub extern "C" fn obs_ptr() -> *mut f32 {
    addr_of_mut!(OBS) as *mut f32
}

#[no_mangle]
pub extern "C" fn act_ptr() -> *mut f32 {
    addr_of_mut!(ACT) as *mut f32
}

#[inline(always)]
fn fabs(x: f32) -> f32 {
    if x < 0.0 {
        -x
    } else {
        x
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

#[inline(always)]
fn wrap01(mut p: f32) -> f32 {
    while p >= 1.0 {
        p -= 1.0;
    }
    while p < 0.0 {
        p += 1.0;
    }
    p
}

/// Triangle wave in [-1, 1] over phase p in [0, 1). No trig required.
#[inline(always)]
fn tri(p: f32) -> f32 {
    4.0 * fabs(wrap01(p) - 0.5) - 1.0
}

const PHASE_STEP: f32 = 0.018; // gait frequency
const HULL_KP: f32 = 2.50; // hull angle stiffness
const HULL_KD: f32 = 0.40; // hull angular-velocity damping
const HIP_AMP: f32 = 0.60; // hip swing amplitude
const KNEE_AMP: f32 = 0.55; // knee swing amplitude

#[no_mangle]
pub extern "C" fn calculate_correction() {
    unsafe {
        let o = &*addr_of_mut!(OBS);
        let hull_angle = o[0];
        let hull_ang_vel = o[1];

        // Advance gait phase (legs driven in antiphase).
        let mut phase = *addr_of_mut!(PHASE);
        phase = wrap01(phase + PHASE_STEP);
        *addr_of_mut!(PHASE) = phase;

        // PD term that keeps the hull level; injected into both hips.
        let hull_corr = -HULL_KP * hull_angle - HULL_KD * hull_ang_vel;

        let hip1 = hull_corr + HIP_AMP * tri(phase);
        let knee1 = KNEE_AMP * tri(phase + 0.25);
        let hip2 = hull_corr + HIP_AMP * tri(phase + 0.5);
        let knee2 = KNEE_AMP * tri(phase + 0.75);

        let a = &mut *addr_of_mut!(ACT);
        a[0] = clamp(hip1, -1.0, 1.0);
        a[1] = clamp(knee1, -1.0, 1.0);
        a[2] = clamp(hip2, -1.0, 1.0);
        a[3] = clamp(knee2, -1.0, 1.0);
    }
}
