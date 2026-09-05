//! 2D top-down walker integration.
//!
//! Turns an 8-way boolean direction input into a unit walk vector,
//! scales by speed + dt, applies to a `DVec2` position, then hands off
//! to a caller-supplied collision resolver. Optional `speed_mult` and
//! `wobble` hooks let games layer buffs (drink effects, terrain
//! modifiers) on top without the engine needing to know about them.
//!
//! Rotation is set from `aim_world - pos` when non-zero; otherwise the
//! caller's existing rotation is preserved.
//!
//! Ships one thing: [`integrate_walker`]. Game-side wrappers (e.g.
//! `void_sim::character::integrate`) build the `WalkParams` and pass a
//! closure that calls their own tile-collision function.

use glam::DVec2;

/// Input parameters for one walker integration step.
///
/// The `n`/`s`/`e`/`w` bits are OR-summed into a raw vector, then
/// normalised so diagonals don't move faster. `speed` (m/s) times `dt`
/// (s) gives the step distance; `sprint` triples it (`SPRINT_MULT`);
/// `speed_mult` is a scalar layered on top of the base — e.g. drink
/// buffs — and `wobble` (m/s) is a perpendicular drift added to the
/// walk vector each frame.
#[derive(Clone, Copy, Debug)]
pub struct WalkParams {
    pub n:          bool,
    pub s:          bool,
    pub e:          bool,
    pub w:          bool,
    pub sprint:     bool,
    /// Base walking speed in m/s. `speed_mult` and sprint stack on this.
    pub speed:      f32,
    /// Optional multiplier on `speed`. Pass 1.0 when unused.
    pub speed_mult: f32,
    /// Optional perpendicular drift in m/s. Pass 0.0 when unused.
    pub wobble:     f32,
}

impl Default for WalkParams {
    fn default() -> Self {
        Self { n: false, s: false, e: false, w: false, sprint: false,
               speed: 0.0, speed_mult: 1.0, wobble: 0.0 }
    }
}

impl WalkParams {
    /// True if any of the four cardinal bits is set.
    pub fn any(self) -> bool { self.n | self.s | self.e | self.w }
}

/// Multiplier applied to `WalkParams.speed` when `sprint` is set.
pub const SPRINT_MULT: f32 = 3.0;

/// One walker integration step.
///
/// Reads `params` for the direction bits + speed knobs, integrates
/// `pos` by `speed * dt`, orients `rot` toward `aim_world`, then calls
/// `collide` to resolve overlap against level geometry.
///
/// Returns `true` if the walker was moving this tick (any direction
/// bit set) — game code typically forwards this into a `walking`
/// animation flag.
pub fn integrate_walker<C>(
    pos:        &mut DVec2,
    rot:        &mut f32,
    aim_world:  DVec2,
    dt:         f32,
    params:     WalkParams,
    collide:    C,
) -> bool
where
    C: FnOnce(&mut DVec2),
{
    let mut dx = 0.0_f64;
    let mut dy = 0.0_f64;
    if params.n { dy += 1.0; }
    if params.s { dy -= 1.0; }
    if params.e { dx += 1.0; }
    if params.w { dx -= 1.0; }
    let len = (dx * dx + dy * dy).sqrt();
    if len > 0.0 {
        let inv = 1.0 / len;
        let base = params.speed * params.speed_mult.max(0.0);
        let speed = if params.sprint { base * SPRINT_MULT } else { base };
        let step = speed as f64 * dt as f64;
        let fx = dx * inv;
        let fy = dy * inv;
        pos.x += fx * step;
        pos.y += fy * step;
        // Sideways wobble: perpendicular to the walk direction so it
        // reads as the character lurching side-to-side while walking,
        // not sliding diagonally. (-fy, fx) is forward rotated 90° CCW.
        if params.wobble != 0.0 {
            let wstep = params.wobble as f64 * dt as f64;
            pos.x += -fy * wstep;
            pos.y +=  fx * wstep;
        }
    }
    let walking = params.any();

    let face = aim_world - *pos;
    if face.length_squared() > 0.0 {
        *rot = face.y.atan2(face.x) as f32;
    }

    collide(pos);
    walking
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagonal_normalises() {
        let mut p = DVec2::ZERO;
        let mut r = 0.0f32;
        let params = WalkParams { n: true, e: true, speed: 3.0, ..Default::default() };
        integrate_walker(&mut p, &mut r, DVec2::new(10.0, 0.0), 1.0, params, |_| {});
        let expected = 3.0_f64 / 2.0_f64.sqrt();
        assert!((p.x - expected).abs() < 1e-6);
        assert!((p.y - expected).abs() < 1e-6);
    }

    #[test]
    fn idle_does_not_move() {
        let mut p = DVec2::new(1.0, 2.0);
        let mut r = 0.5f32;
        let params = WalkParams { speed: 3.0, ..Default::default() };
        let walking = integrate_walker(&mut p, &mut r, DVec2::ZERO, 0.1, params, |_| {});
        assert!(!walking);
        assert!((p - DVec2::new(1.0, 2.0)).length() < 1e-9);
    }

    #[test]
    fn sprint_triples_step() {
        let mut p1 = DVec2::ZERO; let mut r = 0.0f32;
        let mut p2 = DVec2::ZERO;
        let walk = WalkParams { e: true, speed: 3.0, ..Default::default() };
        let sprint = WalkParams { e: true, speed: 3.0, sprint: true, ..Default::default() };
        integrate_walker(&mut p1, &mut r, DVec2::new(1.0, 0.0), 1.0, walk, |_| {});
        integrate_walker(&mut p2, &mut r, DVec2::new(1.0, 0.0), 1.0, sprint, |_| {});
        assert!((p2.x - p1.x * 3.0).abs() < 1e-6);
    }

    #[test]
    fn collision_closure_runs() {
        let mut p = DVec2::new(5.0, 0.0);
        let mut r = 0.0f32;
        let params = WalkParams { speed: 0.0, ..Default::default() };
        integrate_walker(&mut p, &mut r, DVec2::ZERO, 1.0, params, |pos| { pos.x = 99.0; });
        assert!((p.x - 99.0).abs() < 1e-9);
    }
}
