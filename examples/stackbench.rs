//! How well does a stack of boxes actually stand up?
//!
//! A measuring stick, not shipped behaviour. Stacking is the hardest
//! thing a contact solver does — every layer loads the contacts beneath
//! it, and the error compounds downward — so it is the honest test of
//! whether collision response works, and a single number per run is
//! easier to argue with than a video.
//!
//! Four numbers per stack height:
//!
//! * **sink** — worst deviation of any crate from where it should rest.
//!   A stack of `n` 1 m crates should sit at 0.5, 1.5, 2.5 … metres; a
//!   sink near `n-1` means the whole pile passed through itself and
//!   landed in a heap on the floor.
//! * **vel** — the fastest crate still moving after settling. A settled
//!   stack should be at zero.
//! * **pen** — the deepest remaining overlap. The solver deliberately
//!   leaves `PENETRATION_SLOP` (5 mm); much more than that is compression.
//! * **asleep** — how many crates the sleep system managed to retire.
//!
//! Run with:
//!     cargo run --release --example stackbench
//!
//! # The baseline this was written to measure
//!
//! ```text
//! height  sink(m)   vel(m/s)   pen(m)   asleep
//!      1   0.0037    0.0000    0.0050    1/1
//!      2   0.0182    0.0501    0.0137    0/2
//!      3   2.0006    0.0000    0.0050    3/3
//!      5   3.9997    0.0000    0.0050    5/5
//!      8   7.0027    0.0040    0.0084    2/8
//! ```
//!
//! One and two crates hold. **Three or more collapse completely** — every
//! crate ends at z ≈ 0.5, having fallen through the others.
//!
//! The per-tick trace shows why, and it is not slow convergence. The
//! bottom crate oscillates on alternating ticks, and the gap to the one
//! above swings 0.936 / 1.010 around its 1.0 resting value — so on odd
//! ticks the pair genuinely separates, `obb_vs_obb` returns `None`, and
//! four contact points vanish; on even ticks they re-overlap by 65 mm.
//! The amplitude *grows*: 17 mm at tick 10, 52 mm by tick 47.
//!
//! Neither knob helps, which is what rules out tuning as the answer:
//! `SOLVER_ITERATIONS` at 4, 8, 16 and 32 all collapse identically, and
//! so does `BAUMGARTE` at 0.1 through 0.6. The cause is structural — the
//! solver starts every contact impulse from zero every tick, so a resting
//! stack must rediscover its own support force from scratch, and with no
//! memory of the previous tick it overshoots and rings.
//!
//! # What accumulating impulses changed
//!
//! The solver now accumulates each contact point's impulse across the
//! iterations of a tick and clamps the *total* rather than the increment,
//! so a point that over-corrects can be partly undone by the next
//! iteration instead of being skipped by a separating early-out.
//!
//! That did not move the numbers above at four iterations. What it did
//! was make the solver **convergent**, which shows up as soon as the
//! iteration count is raised — something that did nothing whatsoever
//! before:
//!
//! ```text
//! n=3 sink      4 iters   64 iters
//!   before       2.00       2.00
//!   after        2.00       0.03
//! ```
//!
//! So what is left is a convergence *rate* problem rather than a
//! stability one, and 64 iterations a tick is not a shippable answer.
//! Warm starting across ticks is: a resting stack would begin each tick
//! already holding last tick's support impulses, so four iterations only
//! has to absorb the change rather than rediscover the whole answer.
use glam::{DVec3, Quat};
use void_engine::collision::narrow3d;
use void_engine::components::{Transform3D, Velocity3D};
use void_engine::physics3d::{self, BodyRef, Material3D, RigidBody};

const H: f64 = 0.5;

fn run(n: usize, ticks: usize) -> (f64, f64, f64, usize) {
    let fh = [12.0, 12.0, 0.5];
    let mut bodies = vec![RigidBody::static_body()
        .with_material(Material3D { restitution: 0.0, friction: 0.8 })];
    let mut tf = vec![Transform3D::at(DVec3::new(0.0, 0.0, -0.5))];
    let mut vel = vec![Velocity3D::default()];
    for i in 0..n {
        bodies.push(
            RigidBody::box3d(1.0, [H as f32, H as f32, H as f32])
                .with_material(Material3D { restitution: 0.0, friction: 0.8 }),
        );
        // Dropped from just above its resting height, squarely.
        tf.push(Transform3D::at(DVec3::new(0.0, 0.0, H + i as f64 * 2.0 * H + 0.02)));
        vel.push(Velocity3D::default());
    }
    let dt = 1.0 / 60.0;
    for _ in 0..ticks {
        {
            let mut r = refs(&mut bodies, &mut tf, &mut vel);
            physics3d::step(&mut r, physics3d::GRAVITY, dt);
        }
        let contacts = build(&bodies, &tf, fh);
        {
            let mut r = refs(&mut bodies, &mut tf, &mut vel);
            physics3d::solver::solve(&mut r, &contacts, dt as f64);
            physics3d::update_sleep_all(&mut r, dt);
        }
    }
    // Quality metrics.
    let mut worst_sink = 0.0f64;
    for i in 0..n {
        let want = H + i as f64 * 2.0 * H;
        worst_sink = worst_sink.max((tf[i + 1].pos.z - want).abs());
    }
    let worst_v = (1..=n).map(|i| vel[i].linear.length()).fold(0.0, f64::max);
    let contacts = build(&bodies, &tf, fh);
    let worst_pen = contacts.iter().map(|c| c.penetration).fold(0.0, f64::max);
    let asleep = (1..=n).filter(|&i| bodies[i].sleeping).count();
    (worst_sink, worst_v, worst_pen, asleep)
}

fn refs<'a>(
    b: &'a mut [RigidBody],
    t: &'a mut [Transform3D],
    v: &'a mut [Velocity3D],
) -> Vec<BodyRef<'a>> {
    b.iter_mut()
        .zip(t.iter_mut())
        .zip(v.iter_mut())
        .map(|((body, transform), velocity)| BodyRef { body, transform, velocity })
        .collect()
}

fn build(bodies: &[RigidBody], tf: &[Transform3D], fh: [f64; 3]) -> Vec<physics3d::Contact> {
    let half = [H, H, H];
    let mut out = Vec::new();
    for a in 0..bodies.len() {
        for b in (a + 1)..bodies.len() {
            if !bodies[a].kind.is_dynamic() && !bodies[b].kind.is_dynamic() {
                continue;
            }
            let (ia, ib) = if bodies[a].kind.is_dynamic() { (a, b) } else { (b, a) };
            let ha = if ia == 0 { fh } else { half };
            let hb = if ib == 0 { fh } else { half };
            if let Some((nrm, p)) = narrow3d::obb_vs_obb(
                tf[ia].pos, ha, tf[ia].rot, tf[ib].pos, hb, tf[ib].rot,
            ) {
                for (pt, d) in narrow3d::obb_contact_manifold(
                    tf[ia].pos, ha, tf[ia].rot, tf[ib].pos, hb, tf[ib].rot, nrm, p,
                ) {
                    out.push(physics3d::Contact {
                        a: ia,
                        b: ib,
                        normal: nrm,
                        penetration: d,
                        point: pt,
                    });
                }
            }
        }
    }
    out
}

fn main() {
    println!("{:>6} {:>10} {:>10} {:>10} {:>8}", "height", "sink(m)", "vel(m/s)", "pen(m)", "asleep");
    for n in [1usize, 2, 3, 5, 8] {
        let (s, v, p, a) = run(n, 1800);
        println!("{n:>6} {s:>10.4} {v:>10.4} {p:>10.4} {a:>4}/{n}");
    }
    let _ = Quat::IDENTITY;
}
