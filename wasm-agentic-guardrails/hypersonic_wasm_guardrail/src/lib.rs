#![no_std]
//! Hypersonic interceptor structural-load safety envelope (G-load clamp).
//!
//! The hazard: at high velocity, maximum fin/torque deflection produces excessive
//! dynamic pressure / G-loading and the airframe fails structurally. This envelope
//! is a deterministic predicate that bounds the commanded load.
//!
//! Linear-memory model:
//!   OBS[6]    = [speed, pitch_rate, yaw_rate, tgt_x, tgt_y, tgt_z]   host -> wasm
//!   ACT_IN[2] = [pitch_cmd, yaw_cmd]  (agent's aggressive command, [-1,1])  host -> wasm
//!   ACT[2]    = [pitch_cmd, yaw_cmd]  (structurally safe command)    wasm -> host
//!   STATUS[2] = per-channel verdict: 0.0 = passthrough, 1.0 = clamped   wasm -> host
//!
//! Load proxy (per channel):  Load = speed^2 * |command|
//! If Load exceeds the structural G-limit, the command is clamped to the maximum
//! deflection that keeps Load == G_LIMIT, preserving the agent's intended sign.

use core::panic::PanicInfo;
use core::ptr::addr_of_mut;

#[panic_handler]
fn panic(_: &PanicInfo) -> ! {
    loop {}
}

static mut OBS: [f32; 6] = [0.0; 6];
static mut ACT_IN: [f32; 2] = [0.0; 2];
static mut ACT: [f32; 2] = [0.0; 2];
static mut STATUS: [f32; 2] = [0.0; 2];

/// Structural G-load limit in proxy units (speed^2 * command). Runtime-tunable.
static mut G_LIMIT: f32 = 18000.0;

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

/// Runtime control of the structural G-load limit (proxy units).
#[no_mangle]
pub extern "C" fn set_g_limit(v: f32) {
    unsafe {
        *addr_of_mut!(G_LIMIT) = v;
    }
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

/// Evaluate the agent's command against the structural envelope and clamp if needed.
#[no_mangle]
pub extern "C" fn calculate_correction() {
    unsafe {
        let o = &*addr_of_mut!(OBS);
        let speed = o[0];
        let v2 = speed * speed;
        let limit = *addr_of_mut!(G_LIMIT);

        let inp = &*addr_of_mut!(ACT_IN);
        let a = &mut *addr_of_mut!(ACT);
        let s = &mut *addr_of_mut!(STATUS);

        let mut i = 0usize;
        while i < 2 {
            // The actuator can never be driven beyond its physical deflection range.
            let cmd = clamp(inp[i], -1.0, 1.0);
            let load = v2 * fabs(cmd);

            // Below a trivial dynamic-pressure floor there is no structural risk.
            if v2 > 1.0 && load > limit {
                // Maximum |command| that keeps Load == G_LIMIT at this speed.
                let safe_mag = clamp(limit / v2, 0.0, 1.0);
                a[i] = if cmd >= 0.0 { safe_mag } else { -safe_mag };
                s[i] = 1.0;
            } else {
                a[i] = cmd;
                s[i] = 0.0;
            }
            i += 1;
        }
    }
}
