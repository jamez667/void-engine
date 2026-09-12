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

/// Largest distance a single collision resolution may span, in world
/// units, before [`integrate_walker`] splits the move.
///
/// Tile collision resolves by *overlap push*, not by sweeping the
/// movement segment — so a step that lands past a wall's far face
/// overlaps nothing and nothing pushes back. Measured threshold: a
/// walker tunnels when one tick's displacement reaches the tile size,
/// exactly. Through `SPRINT_MULT`, that is 10 m/s of base speed at the
/// 30 Hz headless default and 20 m/s at the 60 Hz windowed one — a
/// vehicle, a dash, or knockback reaches it on a plausible number.
///
/// 0.5 world units is half a 1 m tile, so a move is resolved at least
/// twice per tile crossed. A caller whose tiles are smaller than 1 m
/// should substep further itself.
pub const MAX_COLLIDE_STEP: f64 = 0.5;

/// One walker integration step.
///
/// Reads `params` for the direction bits + speed knobs, integrates
/// `pos` by `speed * dt`, orients `rot` toward `aim_world`, then calls
/// `collide` to resolve overlap against level geometry.
///
/// Returns `true` if the walker was moving this tick (any direction
/// bit set) — game code typically forwards this into a `walking`
/// animation flag.
///
/// # Fast walkers are moved in pieces
///
/// `collide` is called once per sub-step of at most [`MAX_COLLIDE_STEP`]
/// rather than once at the end, because an overlap-push resolver cannot
/// see a wall the walker jumped clean over. It takes `FnMut` for that
/// reason; it was `FnOnce` until 2026-09-11, which made substepping
/// impossible to express.
///
/// A walker moving less than `MAX_COLLIDE_STEP` in a tick — every
/// human-scale character, at either tick rate — takes exactly one
/// sub-step and pays nothing for this.
pub fn integrate_walker<C>(
    pos:        &mut DVec2,
    rot:        &mut f32,
    aim_world:  DVec2,
    dt:         f32,
    params:     WalkParams,
    mut collide: C,
) -> bool
where
    C: FnMut(&mut DVec2),
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
        // Wobble rides along with the forward motion, so it is split the
        // same way rather than applied once at the end — otherwise a
        // substepped walker would get its whole sideways lurch after the
        // last collision check.
        let wstep = if params.wobble != 0.0 {
            params.wobble as f64 * dt as f64
        } else {
            0.0
        };

        // How many pieces the move needs. Driven by total displacement,
        // including wobble, since either component can cross a wall.
        let span = (step.abs() + wstep.abs()).max(0.0);
        let pieces = if span > MAX_COLLIDE_STEP {
            (span / MAX_COLLIDE_STEP).ceil().min(64.0) as u32
        } else {
            1
        };
        let frac = 1.0 / pieces as f64;

        for _ in 0..pieces {
            pos.x += fx * step * frac;
            pos.y += fy * step * frac;
            if wstep != 0.0 {
                // Perpendicular to the walk direction so it reads as the
                // character lurching side-to-side while walking, not
                // sliding diagonally. (-fy, fx) is forward rotated 90° CCW.
                pos.x += -fy * wstep * frac;
                pos.y +=  fx * wstep * frac;
            }
            collide(pos);
        }
    } else {
        // Stationary walkers still resolve: a wall may have moved, or the
        // walker may have been spawned inside one.
        collide(pos);
    }
    let walking = params.any();

    // Facing is computed after the move, so the walker looks from where it
    // ended up rather than where it began.
    let face = aim_world - *pos;
    if face.length_squared() > 0.0 {
        *rot = face.y.atan2(face.x) as f32;
    }

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

    // ── substepping ──────────────────────────────────────────────────
    //
    // Collision here is an overlap push, so it cannot see a wall the
    // walker jumped clean over. These pin that a fast walker is moved in
    // pieces small enough for the resolver to catch.

    /// A wall the walker would have cleared in one step must stop it.
    ///
    /// Against the pre-substepping code this fails: the whole 1.2 m move
    /// landed past the far face, nothing overlapped, and the walker ended
    /// up on the other side. 1.2 m/tick is 12 m/s sprinting at 30 Hz —
    /// the engine's own headless default.
    #[test]
    fn a_fast_walker_does_not_cross_a_solid_wall() {
        // A 1 m wall spanning x in [1.0, 2.0]. The resolver only knows how
        // to push a walker back out of it.
        let push_out = |pos: &mut DVec2| {
            if pos.x > 1.0 && pos.x < 2.0 {
                pos.x = 1.0;
            }
        };

        let mut p = DVec2::new(0.9, 0.0);
        let mut r = 0.0f32;
        // 1.2 m in one tick, straight at the wall.
        let params = WalkParams { e: true, speed: 1.2, ..Default::default() };
        integrate_walker(&mut p, &mut r, DVec2::new(99.0, 0.0), 1.0, params, push_out);

        assert!(
            p.x <= 1.0 + 1e-9,
            "walker tunnelled through the wall to x = {}",
            p.x,
        );
    }

    /// The common case must not pay for the uncommon one: a human-scale
    /// walker crosses no sub-step boundary and the resolver runs once.
    #[test]
    fn a_slow_walker_takes_exactly_one_substep() {
        let mut calls = 0u32;
        let mut p = DVec2::ZERO;
        let mut r = 0.0f32;
        // 0.1 m this tick, well under MAX_COLLIDE_STEP.
        let params = WalkParams { e: true, speed: 0.1, ..Default::default() };
        integrate_walker(&mut p, &mut r, DVec2::new(1.0, 0.0), 1.0, params, |_| {
            calls += 1;
        });
        assert_eq!(calls, 1, "a short move must not be split");
    }

    /// A long move is split finely enough that no single piece exceeds the
    /// limit — which is the property that makes the resolver sufficient.
    #[test]
    fn a_long_move_is_split_below_the_limit() {
        let mut steps: Vec<f64> = Vec::new();
        let mut last = 0.0f64;
        let mut p = DVec2::ZERO;
        let mut r = 0.0f32;
        // 8 m in one tick — 16x the limit.
        let params = WalkParams { e: true, speed: 8.0, ..Default::default() };
        integrate_walker(&mut p, &mut r, DVec2::new(99.0, 0.0), 1.0, params, |pos| {
            steps.push(pos.x - last);
            last = pos.x;
        });

        assert!(steps.len() >= 16, "expected at least 16 pieces, got {}", steps.len());
        for (i, s) in steps.iter().enumerate() {
            assert!(
                *s <= MAX_COLLIDE_STEP + 1e-9,
                "piece {i} spanned {s}, over the {MAX_COLLIDE_STEP} limit",
            );
        }
        assert!((p.x - 8.0).abs() < 1e-9, "the total distance must be unchanged: {}", p.x);
    }

    /// Substepping must not change where a walker ends up when nothing is
    /// in the way — the split is invisible to an unobstructed move.
    #[test]
    fn splitting_does_not_change_the_destination() {
        let mut fast = DVec2::ZERO;
        let mut slow = DVec2::ZERO;
        let mut r = 0.0f32;

        let p_fast = WalkParams { e: true, n: true, speed: 9.0, ..Default::default() };
        integrate_walker(&mut fast, &mut r, DVec2::new(99.0, 99.0), 1.0, p_fast, |_| {});

        // The same displacement delivered as ten separate ticks.
        let p_slow = WalkParams { e: true, n: true, speed: 0.9, ..Default::default() };
        for _ in 0..10 {
            integrate_walker(&mut slow, &mut r, DVec2::new(99.0, 99.0), 1.0, p_slow, |_| {});
        }

        // 1e-6, not 1e-9: ten summed `0.9/√2` steps and one `9.0/√2` step
        // differ by ~1.7e-7 in f64 purely from accumulation order. That is
        // float arithmetic, not a divergence in where the walker goes —
        // and a tolerance tight enough to reject it would be testing
        // summation order rather than substepping.
        assert!((fast - slow).length() < 1e-6, "{fast:?} vs {slow:?}");
    }

    /// An idle walker still resolves, so a wall that moved onto it — or a
    /// spawn inside geometry — is pushed out on the next tick.
    #[test]
    fn a_stationary_walker_still_collides() {
        let mut calls = 0u32;
        let mut p = DVec2::new(5.0, 5.0);
        let mut r = 0.0f32;
        let params = WalkParams { speed: 4.0, ..Default::default() };
        integrate_walker(&mut p, &mut r, DVec2::ZERO, 1.0, params, |_| { calls += 1; });
        assert_eq!(calls, 1, "an idle walker must still be resolved once");
    }
}
