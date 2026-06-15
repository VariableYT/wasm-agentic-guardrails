#![no_std]
//! LunarLanderContinuous deterministic safety envelope (true intercept filter).
//!
//! Linear-memory model:
//!   OBS[8]    = [x, y, vx, vy, angle, ang_vel, leg1_contact, leg2_contact]  (host -> wasm)
//!   ACT_IN[2] = chaos / agent-proposed action [main, lateral]               (host -> wasm)
//!   ACT[2]    = safe action emitted to the actuator [main, lateral]         (wasm -> host)
//!   STATUS[2] = per-channel verdict: 0.0 = passthrough, 1.0 = clamped       (wasm -> host)
//!
//! Exports:
//!   compute_baseline()      -> writes the pure PD safe action into ACT (ignores ACT_IN)
//!   calculate_correction()  -> evaluates ACT_IN against the PD envelope and either
//!                              passes the agent action through or clamps it to the
//!                              safe PD output, per channel.

use core::panic::PanicInfo;
use core::ptr::addr_of_mut;

#[panic_handler]
fn panic(_: &PanicInfo) -> ! {
    loop {}
}

static mut OBS: [f32; 8] = [0.0; 8];
static mut ACT_IN: [f32; 2] = [0.0; 2];
static mut ACT: [f32; 2] = [0.0; 2];
static mut STATUS: [f32; 2] = [0.0; 2];

/// Trust-band half-width (runtime-tunable; see set_delta). Default 0.40.
static mut DELTA: f32 = 0.40;

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

/// Runtime control of the envelope trust-band half-width (action units).
#[no_mangle]
pub extern "C" fn set_delta(v: f32) {
    unsafe {
        *addr_of_mut!(DELTA) = v;
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

/// Deterministic safe baseline: bounded PD hover + attitude hold.
#[inline(always)]
fn baseline(o: &[f32; 8]) -> (f32, f32) {
    let (x, y, vx, vy, angle, ang_vel, leg1, leg2) =
        (o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7]);

    let angle_targ = clamp(x * 0.5 + vx * 1.0, -0.4, 0.4);
    let hover_targ = 0.55 * fabs(x);

    let angle_todo = (angle_targ - angle) * 0.5 - ang_vel * 1.0;
    let hover_todo = (hover_targ - y) * 0.5 - vy * 0.5;

    let mut main = hover_todo * 20.0 - 1.0;
    let mut lateral = -angle_todo * 20.0;

    if leg1 > 0.0 || leg2 > 0.0 {
        lateral = 0.0;
        main -= 0.3;
    }

    (clamp(main, -1.0, 1.0), clamp(lateral, -1.0, 1.0))
}

/// Pure PD safe action (Condition B: WASM-only).
#[no_mangle]
pub extern "C" fn compute_baseline() {
    unsafe {
        let o = &*addr_of_mut!(OBS);
        let (m, l) = baseline(o);
        let a = &mut *addr_of_mut!(ACT);
        a[0] = m;
        a[1] = l;
    }
}

/// True intercept (Condition C): filter the agent action through the PD envelope.
#[no_mangle]
pub extern "C" fn calculate_correction() {
    unsafe {
        let o = &*addr_of_mut!(OBS);
        let (bm, bl) = baseline(o);
        let base = [bm, bl];

        let inp = &*addr_of_mut!(ACT_IN);
        let a = &mut *addr_of_mut!(ACT);
        let s = &mut *addr_of_mut!(STATUS);
        let delta = *addr_of_mut!(DELTA);

        let mut i = 0usize;
        while i < 2 {
            // The agent can never exceed the physical actuator bounds.
            let proposed = clamp(inp[i], -1.0, 1.0);
            let diff = proposed - base[i];

            if fabs(diff) <= delta {
                // Inside the trust envelope: honor the agent's intent.
                a[i] = proposed;
                s[i] = 0.0;
            } else {
                // Destructive deviation: override with the safe PD command.
                a[i] = base[i];
                s[i] = 1.0;
            }
            i += 1;
        }
    }
}
