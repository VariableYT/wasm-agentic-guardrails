#![no_std]
//! CartPole-v1 deterministic safety envelope.
//! State (linear memory): OBS[4] = [cart_x, cart_v, pole_theta, pole_omega]
//!                        ACT[1] = discrete action {0.0 = push left, 1.0 = push right}

use core::panic::PanicInfo;
use core::ptr::addr_of_mut;

#[panic_handler]
fn panic(_: &PanicInfo) -> ! {
    loop {}
}

static mut OBS: [f32; 4] = [0.0; 4];
static mut ACT: [f32; 1] = [0.0; 1];

/// Pointer into linear memory where the host writes the 4 observation floats.
#[no_mangle]
pub extern "C" fn obs_ptr() -> *mut f32 {
    addr_of_mut!(OBS) as *mut f32
}

/// Pointer into linear memory where the host reads the 1 action float.
#[no_mangle]
pub extern "C" fn act_ptr() -> *mut f32 {
    addr_of_mut!(ACT) as *mut f32
}

// Pole-stabilizing full-state feedback gains (LQR-style, hand-tuned).
const K_X: f32 = 0.30; // cart position
const K_V: f32 = 0.80; // cart velocity
const K_THETA: f32 = 8.00; // pole angle (dominant term)
const K_OMEGA: f32 = 1.50; // pole angular velocity

/// Deterministic correction: maps continuous control law -> bang-bang discrete action.
#[no_mangle]
pub extern "C" fn calculate_correction() {
    unsafe {
        let o = &*addr_of_mut!(OBS);
        let u = K_X * o[0] + K_V * o[1] + K_THETA * o[2] + K_OMEGA * o[3];
        let a = &mut *addr_of_mut!(ACT);
        // u > 0  => pole is tipping right => accelerate cart right (action 1).
        a[0] = if u > 0.0 { 1.0 } else { 0.0 };
    }
}
