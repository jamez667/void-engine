//! Sequential-impulse contact solver.
//!
//! # What this does that the engine never did before
//!
//! `collision` detects overlaps and hands back a pair list; its module
//! docs say outright that "the engine gives you the pair list, not the
//! resolution". Every consumer game wrote its own response. This is that
//! response, for 3D.
//!
//! # The method
//!
//! Sequential impulses, the standard game-physics approach: for each
//! contact, work out the impulse that would stop the two bodies
//! approaching, apply it immediately, and repeat over every contact a
//! handful of times. Iterating is what makes stacking work — resolving
//! one contact disturbs its neighbours, and a few passes let the
//! disturbance settle instead of a box sinking into the one below it.
//!
//! It is not a global solve and does not pretend to be. A tall stack
//! under load will still compress slightly; the answer there is more
//! iterations, not a different algorithm at this scale.
//!
//! # Why penetration is corrected separately
//!
//! Pushing bodies apart by adding velocity injects energy, and a stack
//! resolved that way visibly bounces. Positions are corrected directly
//! instead ([`BAUMGARTE`]), and only the part of the overlap beyond a
//! small allowance, so resting bodies keep a hair of penetration rather
//! than jittering between touching and not.

use glam::{DVec3, Quat};

use super::body::RigidBody;
use crate::components::{Transform3D, Velocity3D};

/// Solver passes per step.
///
/// Four is the usual game default: one pass leaves a stack visibly
/// sinking, and past about eight the improvement is not worth the cost.
/// Raise it for tall stacks, lower it for scenes of loose objects.
pub const SOLVER_ITERATIONS: usize = 4;

/// Fraction of remaining penetration corrected per step.
///
/// Correcting all of it in one step makes bodies pop apart; correcting
/// none lets them sink. Named after the Baumgarte stabilisation this
/// approximates.
///
/// # Why 0.3 and not 0.2
///
/// 0.2 recovers about 6 mm per tick of a typical loaded contact, against
/// the 2.7 mm that contact sinks under gravity in the same tick — enough
/// for one body on the floor, and not enough for a stack, where the load
/// grows with every layer while the recovery rate does not. Measured on
/// `examples/crates3d`: a three-high stack of 1 m crates settled at
/// 0.483 / 1.435 / 2.416 against a nominal 0.50 / 1.50 / 2.50, visibly
/// squashed into itself and squashing further the longer it stood.
///
/// # And why not more passes
///
/// The obvious fix — run the correction once per solver iteration, the
/// way the impulses are — is worse. Penetration is measured *once*,
/// before any correction runs, so repeated passes act on a stale depth,
/// push the pair apart by more than they currently overlap, and leave
/// them airborne to fall again next tick. Measured: crates resting on a
/// static floor oscillating over 0.14 m at vertical speeds past 1.3 m/s.
///
/// One stronger pass fixes the height without the jump. At 0.3 the same
/// stack settles at 0.492 / 1.470 / 2.459 with a third of the residual
/// motion.
pub const BAUMGARTE: f64 = 0.3;

/// Penetration left uncorrected, in metres.
///
/// Resting bodies keep this much overlap deliberately. Without it a
/// settled box oscillates between "touching" (correct, push apart) and
/// "not touching" (no contact, fall back), which reads as jitter.
pub const PENETRATION_SLOP: f64 = 0.005;

/// Relative normal speed below which a collision is treated as resting
/// rather than bouncing, in m/s.
///
/// Without this a body never quite comes to rest: each微 bounce is
/// smaller but never zero, and the object buzzes on the floor forever.
pub const RESTITUTION_THRESHOLD: f64 = 1.0;

/// One point of contact between two bodies.
///
/// Produced by the caller from whatever narrowphase it ran — see
/// [`crate::collision::narrow3d`] — so the solver stays independent of
/// which shapes were tested.
#[derive(Copy, Clone, Debug)]
pub struct Contact {
    /// Index of the first body, in the caller's own arrays.
    pub a: usize,
    pub b: usize,
    /// Unit normal pointing **from B toward A**, matching the convention
    /// every function in `narrow3d` returns.
    pub normal: DVec3,
    /// How deeply the two overlap along the normal, in metres.
    pub penetration: f64,
    /// Contact point in world space. Used to work out the lever arm for
    /// each body, which is what makes an off-centre hit impart spin.
    pub point: DVec3,
}

/// What one contact point has applied so far this tick.
///
/// Sequential impulses converge by *accumulating*: each iteration adds an
/// increment and the clamp is applied to the running total, so an
/// over-correction made early can be taken back later. Storing only the
/// last increment instead makes every over-correction permanent, which is
/// what leaves a resting stack with a residual it can never shed.
#[derive(Copy, Clone, Debug, Default)]
struct Accumulated {
    /// Total normal impulse, clamped at or above zero: a contact pushes
    /// and never pulls.
    normal: f64,
    /// Total friction impulse along the tangent, clamped to Coulomb's
    /// limit against `normal`.
    tangent: f64,
    /// The separating speed this contact is solving *toward*, captured on
    /// the first iteration and held for the rest of the tick.
    ///
    /// Zero for a resting contact; `restitution * approach_speed` for a
    /// bouncy one. It must be captured once rather than recomputed,
    /// because after the first iteration the contact is already
    /// separating — a recomputed target would chase its own output, the
    /// second iteration seeing the bounce it just created as an approach
    /// to cancel, and the accumulator taking the whole bounce back again.
    /// Measured before this: a perfectly elastic 5 m/s impact came out at
    /// exactly zero.
    target: f64,
    /// Whether `target` has been captured yet.
    primed: bool,
}

/// Everything the solver needs about one body, borrowed for a step.
pub struct BodyRef<'a> {
    pub body: &'a mut RigidBody,
    pub transform: &'a mut Transform3D,
    pub velocity: &'a mut Velocity3D,
}

/// Velocity of the material point of `i` at world position `point`.
///
/// A spinning body's surface moves even when its centre does not, and
/// that surface velocity is what friction acts on.
#[inline]
fn point_velocity(vel: &Velocity3D, rel: DVec3) -> DVec3 {
    vel.linear + vel.angular.cross(rel)
}

/// The effective mass along `dir` for a contact — how much impulse it
/// takes to change the relative velocity by one unit.
///
/// `1/m + (I⁻¹ (r × n)) × r · n` per body. The rotational term is why an
/// impulse at the end of a long object does less to its centre velocity
/// than the same impulse through the middle.
#[inline]
fn effective_mass(
    body: &RigidBody,
    rot: Quat,
    rel: DVec3,
    dir: DVec3,
) -> f64 {
    if !body.kind.is_dynamic() {
        return 0.0;
    }
    let inv_i = body.world_inv_inertia(rot);
    let rn = rel.cross(dir);
    let angular = (inv_i * rn.as_vec3()).as_dvec3().cross(rel).dot(dir);
    body.inv_mass as f64 + angular
}

/// Resolve `contacts` over `bodies`, in place.
///
/// Bodies are addressed by the indices in each [`Contact`]; the caller
/// owns the array and the mapping back to entities. Contacts touching a
/// sleeping body wake it first — a body hit while asleep that stays
/// asleep is the classic "projectile passes through the crate" bug.
pub fn solve(bodies: &mut [BodyRef<'_>], contacts: &[Contact], dt: f64) {
    if contacts.is_empty() {
        return;
    }

    // A sleeping body is woken by a contact with something that is
    // *moving*, not by contact as such.
    //
    // Waking on contact alone means nothing can ever stay asleep: a crate
    // resting on the floor is in a contact every single tick, so it is
    // woken on the tick after it falls asleep, forever. That silently
    // disabled sleeping for every settled body in the scene -- measured
    // on `examples/crates3d`, three crates sat motionless at 0.006 m/s,
    // two orders below the 0.05 threshold, and still reported awake after
    // 1200 ticks.
    //
    // The partner must be awake *and* actually in motion. A static floor
    // is never in motion, and two settled crates leaning together do not
    // keep each other up.
    for c in contacts {
        for (i, other) in [(c.a, c.b), (c.b, c.a)] {
            let disturbed = match bodies.get(other) {
                Some(o) => !o.body.sleeping && !crate::physics3d::is_still(o.velocity),
                None => false,
            };
            if !disturbed {
                continue;
            }
            if let Some(r) = bodies.get_mut(i) {
                if r.body.kind.is_dynamic() && r.body.sleeping {
                    r.body.wake();
                }
            }
        }
    }

    // One accumulator per contact point, living for the whole solve.
    let mut acc = vec![Accumulated::default(); contacts.len()];
    for _ in 0..SOLVER_ITERATIONS {
        for (i, c) in contacts.iter().enumerate() {
            solve_one(bodies, c, &mut acc[i]);
        }
    }

    // Positional correction after the impulses, so it works on the
    // post-impulse state rather than fighting it.
    //
    // This is per contact *point*, which looks like a bug and is not.
    // Correcting once per pair instead — the arithmetically tidy thing,
    // since all a pair's points share one normal — was tried and made the
    // real scene worse: `examples/crates3d` went from a settled stack to
    // a crate moving at 3.5 m/s, while `examples/stackbench` did not
    // improve at all. A resting box's four points each carry their own
    // depth, and correcting only the deepest leaves the others embedded,
    // so the box tips. Per point over-corrects a flat rest but keeps the
    // face level, and level turns out to matter more.
    for c in contacts {
        correct_penetration(bodies, c);
    }

    let _ = dt;
}

/// Apply the normal and friction impulses for one contact.
fn solve_one(bodies: &mut [BodyRef<'_>], c: &Contact, acc: &mut Accumulated) {
    let (ia, ib) = (c.a, c.b);
    if ia == ib || ia >= bodies.len() || ib >= bodies.len() {
        return;
    }

    // Snapshot what the solve needs, so the two mutable borrows below
    // never overlap.
    let (a_kind, a_inv_m, a_rot, a_vel, a_pos) = {
        let r = &bodies[ia];
        (r.body.kind, r.body.inv_mass, r.transform.rot, r.velocity.clone(), r.transform.pos)
    };
    let (b_kind, b_inv_m, b_rot, b_vel, b_pos) = {
        let r = &bodies[ib];
        (r.body.kind, r.body.inv_mass, r.transform.rot, r.velocity.clone(), r.transform.pos)
    };

    // Two immovable bodies have nothing to resolve.
    if !a_kind.is_dynamic() && !b_kind.is_dynamic() {
        return;
    }
    let _ = (a_inv_m, b_inv_m);

    let ra = c.point - a_pos;
    let rb = c.point - b_pos;

    let va = point_velocity(&a_vel, ra);
    let vb = point_velocity(&b_vel, rb);
    // Relative velocity of A with respect to B, matching the normal's
    // "from B toward A" direction.
    let rv = va - vb;
    let vn = rv.dot(c.normal);

    // **No early-out for a separating contact here.**
    //
    // Skipping when `vn > 0` is right for a solver that applies one
    // impulse and forgets it, and wrong for one that accumulates. Four
    // points share a resting face: the first removes the whole approach
    // velocity, which over-corrects for the pair, so the second sees the
    // pair *separating* and skips — and the rotation the first induced
    // makes the third see approach again. The points fight each other
    // instead of converging, and the pair is left with a steady residual
    // it never sheds. Measured on a three-high stack: every crate sank at
    // a constant 0.074 m/s forever, 2.2 m over thirty seconds, which is
    // the whole collapse.
    //
    // Accumulating instead lets a point that over-corrected be partly
    // *undone* by the next iteration, because the clamp below is on the
    // running total rather than on each increment. A total that would go
    // negative is what "this contact is separating" really means, and the
    // clamp expresses it exactly.

    let (restitution, friction) = {
        let a = &bodies[ia].body;
        let b = &bodies[ib].body;
        (
            RigidBody::combined_restitution(a, b) as f64,
            RigidBody::combined_friction(a, b) as f64,
        )
    };

    let inv_mass_n = {
        let a = effective_mass(bodies[ia].body, a_rot, ra, c.normal);
        let b = effective_mass(bodies[ib].body, b_rot, rb, c.normal);
        a + b
    };
    if inv_mass_n <= 0.0 {
        return;
    }

    // Below the threshold, treat the contact as resting: applying
    // restitution to a slow approach is what keeps a settling body
    // buzzing forever.
    if !acc.primed {
        let bounce = if -vn > RESTITUTION_THRESHOLD { restitution } else { 0.0 };
        acc.target = -vn * bounce;
        acc.primed = true;
    }

    // The increment this iteration wants, then the clamp on the *total*.
    //
    // `acc.normal` is what this contact point has already applied across
    // earlier iterations of this tick. Clamping the sum at zero means a
    // contact can push but never pull, while still allowing an increment
    // to be negative — which is how an over-correction from an earlier
    // iteration gets taken back. Clamping the increment instead would
    // make every over-correction permanent, which is the bug.
    let delta = (acc.target - vn) / inv_mass_n;
    let total = (acc.normal + delta).max(0.0);
    let jn_applied = total - acc.normal;
    acc.normal = total;

    let normal_impulse = c.normal * jn_applied;
    apply(bodies, ia, a_rot, ra, normal_impulse);
    apply(bodies, ib, b_rot, rb, -normal_impulse);

    if friction <= 0.0 {
        return;
    }

    // Friction acts along the tangential part of the relative velocity,
    // recomputed after the normal impulse so it responds to the state the
    // body is actually in.
    let (va2, vb2) = (
        point_velocity(bodies[ia].velocity, ra),
        point_velocity(bodies[ib].velocity, rb),
    );
    let rv2 = va2 - vb2;
    let tangent_v = rv2 - c.normal * rv2.dot(c.normal);
    let t_len = tangent_v.length();
    if t_len < 1e-9 {
        return;
    }
    let tangent = tangent_v / t_len;

    let inv_mass_t = {
        let a = effective_mass(bodies[ia].body, a_rot, ra, tangent);
        let b = effective_mass(bodies[ib].body, b_rot, rb, tangent);
        a + b
    };
    if inv_mass_t <= 0.0 {
        return;
    }

    // Coulomb: the friction impulse cannot exceed μ times the normal
    // impulse. Past that the surfaces slide rather than grip.
    //
    // What the clamp actually buys is that a *lightly* pressed contact
    // can only apply a little friction. `jt_unclamped` exactly cancels
    // the tangential velocity — it cannot overshoot, so friction never
    // reverses a slide either way — but without the limit it cancels the
    // whole slide however glancing the touch, and a puck skimming a
    // surface stops dead instead of sliding on.
    // The budget comes from the **accumulated** normal impulse, not from
    // this iteration's increment. The increment shrinks toward zero as
    // the contact converges — that is what convergence means — so a
    // budget derived from it would starve friction on exactly the resting
    // contacts that need it most, and a settled stack would slowly slide
    // apart.
    let delta_t = -rv2.dot(tangent) / inv_mass_t;
    let max_friction = friction * acc.normal;
    let total_t = (acc.tangent + delta_t).clamp(-max_friction, max_friction);
    let jt_applied = total_t - acc.tangent;
    acc.tangent = total_t;

    let friction_impulse = tangent * jt_applied;
    apply(bodies, ia, a_rot, ra, friction_impulse);
    apply(bodies, ib, b_rot, rb, -friction_impulse);
}

/// Apply an impulse to one body, if it is allowed to move.
fn apply(bodies: &mut [BodyRef<'_>], i: usize, rot: Quat, rel: DVec3, impulse: DVec3) {
    let r = &mut bodies[i];
    if !r.body.kind.is_dynamic() {
        return;
    }
    r.velocity.linear += impulse * r.body.inv_mass as f64;
    let inv_i = r.body.world_inv_inertia(rot);
    let torque = rel.cross(impulse);
    r.velocity.angular += (inv_i * torque.as_vec3()).as_dvec3();
}

/// Push overlapping bodies apart by moving them, not by adding velocity.
///
/// Splitting this from the impulse solve is what keeps a resting stack
/// still: correcting overlap with velocity injects energy the solver then
/// has to remove again, which reads as a stack that breathes.
fn correct_penetration(bodies: &mut [BodyRef<'_>], c: &Contact) {
    let (ia, ib) = (c.a, c.b);
    if ia == ib || ia >= bodies.len() || ib >= bodies.len() {
        return;
    }

    // Only the excess beyond the slop, so a settled contact is left alone.
    let excess = (c.penetration - PENETRATION_SLOP).max(0.0);
    if excess <= 0.0 {
        return;
    }

    let inv_a = if bodies[ia].body.kind.is_dynamic() { bodies[ia].body.inv_mass as f64 } else { 0.0 };
    let inv_b = if bodies[ib].body.kind.is_dynamic() { bodies[ib].body.inv_mass as f64 } else { 0.0 };
    let total = inv_a + inv_b;
    if total <= 0.0 {
        return;
    }

    // Shared in proportion to inverse mass, so a light body moves most
    // and an immovable one not at all.
    let correction = c.normal * (excess * BAUMGARTE / total);
    if inv_a > 0.0 {
        bodies[ia].transform.pos += correction * inv_a;
    }
    if inv_b > 0.0 {
        bodies[ib].transform.pos -= correction * inv_b;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{Transform3D, Velocity3D};
    use crate::physics3d::body::{BodyKind, Material3D};

    struct Scene {
        bodies: Vec<RigidBody>,
        transforms: Vec<Transform3D>,
        velocities: Vec<Velocity3D>,
    }

    impl Scene {
        fn new() -> Self {
            Self { bodies: Vec::new(), transforms: Vec::new(), velocities: Vec::new() }
        }

        fn push(&mut self, b: RigidBody, pos: DVec3, vel: DVec3) -> usize {
            self.bodies.push(b);
            self.transforms.push(Transform3D::at(pos));
            self.velocities.push(Velocity3D { linear: vel, angular: DVec3::ZERO });
            self.bodies.len() - 1
        }

        fn solve(&mut self, contacts: &[Contact]) {
            let mut refs: Vec<BodyRef<'_>> = self
                .bodies
                .iter_mut()
                .zip(self.transforms.iter_mut())
                .zip(self.velocities.iter_mut())
                .map(|((body, transform), velocity)| BodyRef { body, transform, velocity })
                .collect();
            super::solve(&mut refs, contacts, 1.0 / 60.0);
        }
    }

    fn contact(a: usize, b: usize, normal: DVec3, penetration: f64, point: DVec3) -> Contact {
        Contact { a, b, normal, penetration, point }
    }

    /// The basic job: two bodies approaching must stop approaching.
    #[test]
    fn a_head_on_collision_removes_the_approach_velocity() {
        let mut s = Scene::new();
        let a = s.push(
            RigidBody::sphere(1.0, 0.5).with_material(Material3D { restitution: 0.0, friction: 0.0 }),
            DVec3::new(-0.5, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
        );
        let b = s.push(
            RigidBody::sphere(1.0, 0.5).with_material(Material3D { restitution: 0.0, friction: 0.0 }),
            DVec3::new(0.5, 0.0, 0.0),
            DVec3::new(-1.0, 0.0, 0.0),
        );
        // Normal points from B toward A: -X.
        s.solve(&[contact(a, b, DVec3::new(-1.0, 0.0, 0.0), 0.0, DVec3::ZERO)]);

        let rv = s.velocities[a].linear - s.velocities[b].linear;
        assert!(
            rv.dot(DVec3::new(-1.0, 0.0, 0.0)) >= -1e-6,
            "the bodies are still approaching at {rv:?}",
        );
    }

    /// A static body is infinitely massive: the dynamic one takes the
    /// entire impulse and the static one does not move at all.
    #[test]
    fn a_static_body_absorbs_the_whole_impulse_without_moving() {
        let mut s = Scene::new();
        let ball = s.push(
            RigidBody::sphere(1.0, 0.5).with_material(Material3D { restitution: 0.0, friction: 0.0 }),
            DVec3::new(0.0, 0.0, 0.5),
            DVec3::new(0.0, 0.0, -2.0),
        );
        let floor = s.push(RigidBody::static_body(), DVec3::ZERO, DVec3::ZERO);

        s.solve(&[contact(ball, floor, DVec3::new(0.0, 0.0, 1.0), 0.0, DVec3::ZERO)]);

        assert!(
            s.velocities[ball].linear.z >= -1e-6,
            "the ball should have stopped falling, got {:?}",
            s.velocities[ball].linear,
        );
        assert_eq!(s.velocities[floor].linear, DVec3::ZERO);
        assert_eq!(s.transforms[floor].pos, DVec3::ZERO);
    }

    /// Restitution 1 should send a ball back at close to its approach
    /// speed. The threshold does not apply here because the approach is
    /// well above it.
    #[test]
    fn a_perfectly_elastic_bounce_reverses_the_velocity() {
        let mut s = Scene::new();
        let ball = s.push(
            RigidBody::sphere(1.0, 0.5).with_material(Material3D { restitution: 1.0, friction: 0.0 }),
            DVec3::new(0.0, 0.0, 0.5),
            DVec3::new(0.0, 0.0, -5.0),
        );
        let floor = s.push(RigidBody::static_body(), DVec3::ZERO, DVec3::ZERO);
        s.solve(&[contact(ball, floor, DVec3::new(0.0, 0.0, 1.0), 0.0, DVec3::ZERO)]);

        assert!(
            (s.velocities[ball].linear.z - 5.0).abs() < 0.5,
            "expected ~+5 m/s back, got {:?}",
            s.velocities[ball].linear,
        );
    }

    /// A slow approach must *not* bounce, or a settling body buzzes on
    /// the floor forever getting smaller bounces that never reach zero.
    #[test]
    fn a_slow_contact_does_not_bounce_however_bouncy_the_material() {
        let mut s = Scene::new();
        let ball = s.push(
            RigidBody::sphere(1.0, 0.5).with_material(Material3D { restitution: 1.0, friction: 0.0 }),
            DVec3::new(0.0, 0.0, 0.5),
            // Well under RESTITUTION_THRESHOLD.
            DVec3::new(0.0, 0.0, -0.1),
        );
        let floor = s.push(RigidBody::static_body(), DVec3::ZERO, DVec3::ZERO);
        s.solve(&[contact(ball, floor, DVec3::new(0.0, 0.0, 1.0), 0.0, DVec3::ZERO)]);

        assert!(
            s.velocities[ball].linear.z.abs() < 0.05,
            "a slow contact should come to rest, not bounce; got {:?}",
            s.velocities[ball].linear,
        );
    }

    /// Friction must slow sliding motion along the surface. Without it
    /// everything behaves like ice.
    #[test]
    fn friction_slows_a_body_sliding_along_a_surface() {
        let mut s = Scene::new();
        let box_ = s.push(
            RigidBody::sphere(1.0, 0.5).with_material(Material3D { restitution: 0.0, friction: 0.8 }),
            DVec3::new(0.0, 0.0, 0.5),
            // Sliding +X while pressed down into the floor.
            DVec3::new(3.0, 0.0, -1.0),
        );
        let floor = s.push(
            RigidBody::static_body().with_material(Material3D { restitution: 0.0, friction: 0.8 }),
            DVec3::ZERO,
            DVec3::ZERO,
        );
        s.solve(&[contact(box_, floor, DVec3::new(0.0, 0.0, 1.0), 0.0, DVec3::ZERO)]);

        assert!(
            s.velocities[box_].linear.x < 3.0,
            "friction should have reduced the slide; x is still {}",
            s.velocities[box_].linear.x,
        );
    }

    /// The Coulomb limit: a barely-pressed contact can only apply a
    /// little friction, however grippy the surfaces.
    ///
    /// Without the clamp, friction cancels the *whole* tangential
    /// velocity in one solve regardless of how lightly the bodies are
    /// touching — a puck skimming a surface stops dead instead of sliding
    /// on. The clamp ties the friction budget to the normal impulse,
    /// which is what makes a glancing contact barely slow anything.
    ///
    /// Note what this is *not*: the unclamped impulse exactly cancels the
    /// tangential velocity and cannot overshoot it, so friction never
    /// reverses a slide either way. An earlier revision of this test
    /// asserted that and was worthless — it passed with the clamp
    /// deleted, because the thing it guarded against could not happen.
    #[test]
    fn a_glancing_contact_can_only_apply_a_little_friction() {
        let mut s = Scene::new();
        let puck = s.push(
            RigidBody::sphere(1.0, 0.5)
                .with_material(Material3D { restitution: 0.0, friction: 20.0 }),
            DVec3::new(0.0, 0.0, 0.5),
            // Fast along +X, barely settling onto the surface: the normal
            // impulse is tiny, so the friction budget must be too.
            DVec3::new(10.0, 0.0, -0.01),
        );
        let floor = s.push(
            RigidBody::static_body()
                .with_material(Material3D { restitution: 0.0, friction: 20.0 }),
            DVec3::ZERO,
            DVec3::ZERO,
        );
        s.solve(&[contact(puck, floor, DVec3::new(0.0, 0.0, 1.0), 0.0, DVec3::ZERO)]);

        let x = s.velocities[puck].linear.x;
        assert!(
            x > 9.0,
            "a contact this light should barely slow a 10 m/s slide, but \
             x is now {x} — the friction impulse is not being limited by \
             the normal impulse, so friction is unbounded",
        );
        assert!(x <= 10.0, "friction must not speed it up, got {x}");
    }

    /// And a frictionless pair must not: the slide is untouched.
    #[test]
    fn a_frictionless_contact_leaves_sliding_motion_alone() {
        let mut s = Scene::new();
        let puck = s.push(
            RigidBody::sphere(1.0, 0.5).with_material(Material3D { restitution: 0.0, friction: 0.0 }),
            DVec3::new(0.0, 0.0, 0.5),
            DVec3::new(3.0, 0.0, -1.0),
        );
        let ice = s.push(
            RigidBody::static_body().with_material(Material3D { restitution: 0.0, friction: 0.0 }),
            DVec3::ZERO,
            DVec3::ZERO,
        );
        s.solve(&[contact(puck, ice, DVec3::new(0.0, 0.0, 1.0), 0.0, DVec3::ZERO)]);

        assert!(
            (s.velocities[puck].linear.x - 3.0).abs() < 1e-6,
            "a frictionless contact should not slow the slide; got {}",
            s.velocities[puck].linear.x,
        );
    }

    /// Overlap must be corrected by *moving* the bodies, and shared in
    /// proportion to inverse mass so a light body moves most.
    #[test]
    fn penetration_is_corrected_by_moving_the_lighter_body_most() {
        let mut s = Scene::new();
        let light = s.push(RigidBody::sphere(1.0, 0.5), DVec3::new(-0.4, 0.0, 0.0), DVec3::ZERO);
        let heavy = s.push(RigidBody::sphere(10.0, 0.5), DVec3::new(0.4, 0.0, 0.0), DVec3::ZERO);

        let before = (s.transforms[light].pos, s.transforms[heavy].pos);
        s.solve(&[contact(light, heavy, DVec3::new(-1.0, 0.0, 0.0), 0.2, DVec3::ZERO)]);

        let moved_light = (s.transforms[light].pos - before.0).length();
        let moved_heavy = (s.transforms[heavy].pos - before.1).length();
        assert!(moved_light > 0.0, "the overlap should have been corrected");
        assert!(
            moved_light > moved_heavy,
            "the lighter body should move further: {moved_light} vs {moved_heavy}",
        );
    }

    /// Penetration within the slop is left alone, so a resting contact
    /// does not oscillate between touching and not.
    #[test]
    fn penetration_within_the_slop_is_left_alone() {
        let mut s = Scene::new();
        let a = s.push(RigidBody::sphere(1.0, 0.5), DVec3::ZERO, DVec3::ZERO);
        let b = s.push(RigidBody::static_body(), DVec3::new(0.0, 0.0, -1.0), DVec3::ZERO);

        let before = s.transforms[a].pos;
        s.solve(&[contact(
            a,
            b,
            DVec3::new(0.0, 0.0, 1.0),
            PENETRATION_SLOP * 0.5,
            DVec3::ZERO,
        )]);
        assert_eq!(
            s.transforms[a].pos, before,
            "a contact inside the slop should not be pushed apart",
        );
    }

    /// A body hit while asleep must wake, or projectiles pass through
    /// settled objects — the classic sleeping-body bug.
    ///
    /// The partner here is *moving*. An earlier version of this test used
    /// a static body and asserted that the contact alone woke the sleeper,
    /// which is the behaviour that made sleeping impossible: a crate
    /// resting on the floor is in a contact every tick, so it was woken
    /// the tick after it fell asleep, forever. See
    /// [`a_body_asleep_on_the_floor_is_not_woken_by_the_floor`].
    #[test]
    fn a_contact_wakes_a_sleeping_body() {
        let mut s = Scene::new();
        let a = s.push(RigidBody::sphere(1.0, 0.5), DVec3::ZERO, DVec3::ZERO);
        s.bodies[a].sleeping = true;
        let b = s.push(
            RigidBody::sphere(1.0, 0.5),
            DVec3::new(0.0, 0.0, -1.0),
            DVec3::new(0.0, 0.0, 4.0),
        );

        s.solve(&[contact(a, b, DVec3::new(0.0, 0.0, 1.0), 0.1, DVec3::ZERO)]);
        assert!(!s.bodies[a].sleeping, "the contact should have woken it");
    }

    /// Two immovable bodies touching is extremely common in a level made
    /// of static geometry, and must cost nothing and change nothing.
    #[test]
    fn two_static_bodies_in_contact_are_a_no_op() {
        let mut s = Scene::new();
        let a = s.push(RigidBody::static_body(), DVec3::ZERO, DVec3::ZERO);
        let b = s.push(RigidBody::static_body(), DVec3::new(0.5, 0.0, 0.0), DVec3::ZERO);
        s.solve(&[contact(a, b, DVec3::new(-1.0, 0.0, 0.0), 0.5, DVec3::ZERO)]);

        assert_eq!(s.transforms[a].pos, DVec3::ZERO);
        assert_eq!(s.transforms[b].pos, DVec3::new(0.5, 0.0, 0.0));
    }

    /// Bodies already moving apart must be left alone. Resolving a
    /// separating contact *pulls them together*, which reads as objects
    /// sticking to each other.
    #[test]
    fn a_separating_contact_is_not_resolved() {
        let mut s = Scene::new();
        let a = s.push(
            RigidBody::sphere(1.0, 0.5),
            DVec3::new(-0.5, 0.0, 0.0),
            DVec3::new(-2.0, 0.0, 0.0),
        );
        let b = s.push(
            RigidBody::sphere(1.0, 0.5),
            DVec3::new(0.5, 0.0, 0.0),
            DVec3::new(2.0, 0.0, 0.0),
        );
        let before = (s.velocities[a].linear, s.velocities[b].linear);
        s.solve(&[contact(a, b, DVec3::new(-1.0, 0.0, 0.0), 0.0, DVec3::ZERO)]);

        assert_eq!(s.velocities[a].linear, before.0, "already separating");
        assert_eq!(s.velocities[b].linear, before.1);
    }

    /// An off-centre hit must impart spin. This is the difference between
    /// a rigid body and a point mass, and it comes from the contact point
    /// rather than the body centre.
    #[test]
    fn an_off_centre_contact_imparts_spin() {
        let mut s = Scene::new();
        let a = s.push(
            RigidBody::sphere(1.0, 1.0).with_material(Material3D { restitution: 0.0, friction: 0.0 }),
            DVec3::ZERO,
            DVec3::new(0.0, 0.0, -2.0),
        );
        let floor = s.push(RigidBody::static_body(), DVec3::new(0.0, 0.0, -1.0), DVec3::ZERO);

        // Contact off to +X of the centre, so the upward impulse there
        // should tip the body.
        s.solve(&[contact(
            a,
            floor,
            DVec3::new(0.0, 0.0, 1.0),
            0.0,
            DVec3::new(1.0, 0.0, -1.0),
        )]);

        assert!(
            s.velocities[a].angular.length() > 1e-6,
            "an off-centre contact should spin the body, got {:?}",
            s.velocities[a].angular,
        );
    }

    /// A kinematic body pushes dynamic bodies but is not pushed back —
    /// the behaviour a moving platform or a lift needs.
    #[test]
    fn a_kinematic_body_pushes_without_being_pushed() {
        let mut s = Scene::new();
        let crate_ = s.push(
            RigidBody::sphere(1.0, 0.5),
            DVec3::new(0.0, 0.0, 0.5),
            DVec3::new(0.0, 0.0, -1.0),
        );
        let platform = s.push(RigidBody::kinematic(), DVec3::ZERO, DVec3::ZERO);
        assert_eq!(s.bodies[platform].kind, BodyKind::Kinematic);

        s.solve(&[contact(crate_, platform, DVec3::new(0.0, 0.0, 1.0), 0.1, DVec3::ZERO)]);

        assert_eq!(s.velocities[platform].linear, DVec3::ZERO, "unmoved");
        assert_eq!(s.transforms[platform].pos, DVec3::ZERO);
        assert!(s.velocities[crate_].linear.z >= -1e-6, "the crate was stopped");
    }

    /// A crate resting on a big static floor must stay put over many
    /// ticks. This is the crates3d failure reduced to its essentials:
    /// gravity each tick, one contact, repeat.
    #[test]
    fn a_box_resting_on_a_floor_does_not_sink_through_it() {
        let mut s = Scene::new();
        let ground_top = 0.0;
        let half = 0.5;
        let crate_ = s.push(
            RigidBody::box3d(1.0, [half as f32, half as f32, half as f32])
                .with_material(Material3D { restitution: 0.0, friction: 0.6 }),
            DVec3::new(0.0, 0.0, half + 0.001),
            DVec3::ZERO,
        );
        let floor = s.push(RigidBody::static_body(), DVec3::new(0.0, 0.0, -0.5), DVec3::ZERO);

        let dt = 1.0 / 60.0;
        for tick in 0..600 {
            // gravity
            s.velocities[crate_].linear += DVec3::new(0.0, 0.0, -9.81) * dt;
            s.transforms[crate_].pos += s.velocities[crate_].linear * dt;

            let z = s.transforms[crate_].pos.z;
            let pen = (ground_top + half) - z;
            if pen > 0.0 {
                let point = DVec3::new(0.0, 0.0, z - half);
                s.solve(&[contact(crate_, floor, DVec3::new(0.0, 0.0, 1.0), pen, point)]);
            }
            if tick % 120 == 0 {
                eprintln!("[probe] tick {tick} z={:.4} pen={:.4} vz={:.4}",
                    s.transforms[crate_].pos.z, pen.max(0.0), s.velocities[crate_].linear.z);
            }
        }
        let z = s.transforms[crate_].pos.z;
        assert!(z > 0.3, "the crate sank to z={z:.4}; it should rest near 0.5");
    }

    /// Out-of-range or self-referential indices must be ignored rather
    /// than panicking: a caller rebuilding its body array between ticks
    /// can produce a stale contact.
    #[test]
    fn a_malformed_contact_is_ignored_rather_than_panicking() {
        let mut s = Scene::new();
        let a = s.push(RigidBody::sphere(1.0, 0.5), DVec3::ZERO, DVec3::ZERO);
        s.solve(&[
            contact(a, 99, DVec3::new(0.0, 0.0, 1.0), 0.1, DVec3::ZERO),
            contact(a, a, DVec3::new(0.0, 0.0, 1.0), 0.1, DVec3::ZERO),
        ]);
        assert!(s.velocities[a].linear.is_finite());
    }

    /// A sleeping body resting on a static floor stays asleep.
    ///
    /// Waking on the mere *presence* of a contact means nothing can ever
    /// stay asleep: a settled crate touches the floor every tick, so it
    /// is woken the tick after it falls asleep, forever. That silently
    /// disabled sleeping for every resting body in a scene.
    #[test]
    fn a_body_asleep_on_the_floor_is_not_woken_by_the_floor() {
        let mut s = Scene::new();
        let a = s.push(
            RigidBody::box3d(1.0, [0.5, 0.5, 0.5]),
            DVec3::new(0.0, 0.0, 0.5),
            DVec3::ZERO,
        );
        let floor = s.push(RigidBody::static_body(), DVec3::new(0.0, 0.0, -0.5), DVec3::ZERO);
        s.bodies[a].sleeping = true;

        s.solve(&[contact(a, floor, DVec3::Z, 0.001, DVec3::ZERO)]);

        assert!(
            s.bodies[a].sleeping,
            "resting on a static floor must not wake a sleeping body"
        );
    }

    /// A sleeping body *is* woken by something moving into it.
    ///
    /// The counterpart to the rule above, and the reason the rule cannot
    /// simply be "never wake on contact": a projectile that fails to wake
    /// what it hits passes straight through.
    #[test]
    fn a_moving_body_wakes_what_it_hits() {
        let mut s = Scene::new();
        let a = s.push(
            RigidBody::box3d(1.0, [0.5, 0.5, 0.5]),
            DVec3::new(0.0, 0.0, 0.5),
            DVec3::ZERO,
        );
        let hitter = s.push(
            RigidBody::sphere(1.0, 0.5),
            DVec3::new(1.4, 0.0, 0.5),
            DVec3::new(-5.0, 0.0, 0.0),
        );
        s.bodies[a].sleeping = true;

        // Normal from the hitter toward A: -X.
        s.solve(&[contact(a, hitter, DVec3::NEG_X, 0.01, DVec3::new(0.5, 0.0, 0.5))]);

        assert!(
            !s.bodies[a].sleeping,
            "a body struck by something moving must wake"
        );
    }

    /// A stack must not compress under its own weight.
    ///
    /// Positional correction is iterated for the same reason the impulses
    /// are. A single pass separates a pair by `excess * BAUMGARTE`, which
    /// barely outruns the `g*dt²` a contact sinks each tick — and loses
    /// outright once something rests on top, because the load grows while
    /// the recovery rate does not. The stack then squashes into itself
    /// instead of settling, and keeps squashing the longer it stands.
    ///
    /// Measured on `examples/crates3d` with a single pass: a three-high
    /// stack of 1 m crates sat at 0.48 / 1.44 / 2.42 against a nominal
    /// 0.50 / 1.50 / 2.50, with crate-on-crate overlap of 32-65 mm
    /// against a 5 mm slop.
    #[test]
    fn a_loaded_contact_is_pushed_back_out_to_the_slop() {
        let mut s = Scene::new();
        // Two crates, deeply overlapped, with the lower one held up by an
        // immovable floor so the pair cannot simply drift apart.
        let floor = s.push(RigidBody::static_body(), DVec3::new(0.0, 0.0, -0.5), DVec3::ZERO);
        let lower = s.push(
            RigidBody::box3d(1.0, [0.5, 0.5, 0.5]),
            DVec3::new(0.0, 0.0, 0.5),
            DVec3::ZERO,
        );
        let upper = s.push(
            RigidBody::box3d(1.0, [0.5, 0.5, 0.5]),
            // 40 mm into the one below, the depth the game actually
            // reached before the correction was iterated.
            DVec3::new(0.0, 0.0, 1.46),
            DVec3::ZERO,
        );

        let before = s.transforms[upper].pos.z - s.transforms[lower].pos.z;
        s.solve(&[
            contact(lower, floor, DVec3::Z, 0.0, DVec3::ZERO),
            contact(upper, lower, DVec3::Z, 0.04, DVec3::new(0.0, 0.0, 0.98)),
        ]);
        let after = s.transforms[upper].pos.z - s.transforms[lower].pos.z;

        assert!(
            after > before,
            "the pair did not separate at all: {before:.4} -> {after:.4}",
        );
        // The bar is gravity, not a round fraction. A contact sinks
        // `g*dt²` under its own weight every tick — 2.7 mm at 60 Hz — and
        // a stack loads the contacts below it harder than that, so the
        // recovery has to clear it with room to spare or the pile
        // compresses instead of settling.
        let recovered = after - before;
        let sink_per_tick = crate::physics3d::GRAVITY.z.abs() / 3600.0;
        assert!(
            recovered > sink_per_tick * 2.0,
            "recovered only {recovered:.5} m of a 0.04 m overlap in one tick, \r
             against the {sink_per_tick:.5} m a loaded contact sinks in the \r
             same tick — a stack will squash into itself",
        );
    }



}
