#![no_std]
//! Shadow Hand (HandManipulateBlock-v1) joint-limit safety envelope.
//!
//! The hazard: violent random commands drive the 20 finger actuators to their
//! mechanical hard-stops, hyperextending joints and intersecting fingers -- which
//! would destroy the servos on real hardware.
//!
//! The filter works in NORMALISED joint space: the action and the measured joint
//! position are both scaled so that +/-1.0 is the mechanical limit. Per channel it
//! (a) clamps the command into a safe operating band [-SAFE_LIMIT, +SAFE_LIMIT],
//! keeping the servo off the hard-stop, and (b) applies a predictive halt: if the
//! joint's measured position + lookahead*velocity is already past the safe band and
//! the command pushes it further outward, the channel is frozen at its current
//! position (servo halted before hyperextension).
//!
//! Linear-memory model:
//!   OBS[40]    : OBS[0..20]  = normalised joint positions  (+/-1 = limit)
//!                OBS[20..40] = normalised joint velocities (per second)
//!   ACT_IN[20] = agent's proposed normalised command       host -> wasm
//!   ACT[20]    = safe normalised command to the actuators   wasm -> host
//!   STATUS[1]  = number of channels clamped this tick        wasm -> host

use core::panic::PanicInfo;
use core::ptr::addr_of_mut;

#[panic_handler]
fn panic(_: &PanicInfo) -> ! {
    loop {}
}

const N_ACT: usize = 20;
const N_OBS: usize = 40;

static mut OBS: [f32; N_OBS] = [0.0; N_OBS];
static mut ACT_IN: [f32; N_ACT] = [0.0; N_ACT];
static mut ACT: [f32; N_ACT] = [0.0; N_ACT];
static mut STATUS: [f32; 1] = [0.0; 1];

/// Safe operating band as a fraction of each joint's range (1.0 = hard-stop).
static mut SAFE_LIMIT: f32 = 0.90;
/// Velocity lookahead (seconds) for the predictive halt.
static mut LOOKAHEAD: f32 = 0.05;

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
pub extern "C" fn set_safe_limit(v: f32) {
    unsafe {
        *addr_of_mut!(SAFE_LIMIT) = v;
    }
}

#[no_mangle]
pub extern "C" fn set_lookahead(v: f32) {
    unsafe {
        *addr_of_mut!(LOOKAHEAD) = v;
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

/// Evaluate the agent's command against the joint-limit envelope.
#[no_mangle]
pub extern "C" fn calculate_correction() {
    unsafe {
        let o = &*addr_of_mut!(OBS);
        let inp = &*addr_of_mut!(ACT_IN);
        let a = &mut *addr_of_mut!(ACT);
        let s = &mut *addr_of_mut!(STATUS);

        let safe = *addr_of_mut!(SAFE_LIMIT);
        let look = *addr_of_mut!(LOOKAHEAD);

        let mut count: u32 = 0;
        let mut i = 0usize;
        while i < N_ACT {
            let tgt = clamp(inp[i], -1.0, 1.0);
            let p = o[i];
            let v = o[N_ACT + i];
            let predicted = p + look * v;

            // (a) hard operating band keeps the servo off the mechanical stop.
            let mut out = clamp(tgt, -safe, safe);

            // (b) predictive halt: never drive a joint that is already (or about to be)
            //     past the safe band any further toward its hard-stop.
            if predicted > safe && out > p {
                out = p;
            }
            if predicted < -safe && out < p {
                out = p;
            }

            if out != tgt {
                count += 1;
            }
            a[i] = out;
            i += 1;
        }
        s[0] = count as f32;
    }
}
