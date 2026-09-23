//! Rigid-body components: mass, inertia, and what a body is allowed to do.

use glam::{DVec3, Mat3, Quat};

#[cfg(feature = "persist")]
use serde::{Deserialize, Serialize};

/// Smallest mass a body may have, in kilograms.
///
/// A gram. Clamping at `f32::MIN_POSITIVE` instead would let `1/m`
/// overflow to infinity, and an infinite inverse mass means any impulse
/// sends the body off at an unrepresentable speed.
const MIN_MASS: f32 = 0.001;

/// Smallest full extent used to build an inertia tensor, in metres.
///
/// A millimetre. The tensor squares extents and scales by m/12, so a
/// clamp near `f32::MIN_POSITIVE` still underflows that product to zero
/// and puts an infinity in the tensor — which a test caught on a
/// zero-sized box.
const MIN_EXTENT: f32 = 0.001;

/// What moves a body, and what it moves in response to.
#[cfg_attr(feature = "persist", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum BodyKind {
    /// Moved by forces and impulses. Falls, bounces, is pushed.
    #[default]
    Dynamic,
    /// Moved only by whatever sets its transform, but pushes dynamic
    /// bodies out of the way. A lift, a moving platform, a character
    /// controller.
    ///
    /// Treated as infinitely massive by the solver: a dynamic body
    /// bouncing off one takes the whole impulse.
    Kinematic,
    /// Never moves. Level geometry.
    ///
    /// Distinct from `Kinematic` because two statics are never even
    /// tested against each other, which is most of the pair list in a
    /// scene made mostly of walls.
    Static,
}

impl BodyKind {
    /// Whether the solver may move this body.
    pub fn is_dynamic(self) -> bool {
        matches!(self, BodyKind::Dynamic)
    }
}

/// Surface response: how bouncy, how grippy.
///
/// Per body rather than per contact pair. A pair's values are combined
/// when they meet — see [`RigidBody::combined_restitution`] — which is
/// the usual simplification: a real pair table is `n²` entries a game
/// almost never fills in.
#[cfg_attr(feature = "persist", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Material3D {
    /// 0 = dead stop, 1 = bounces to the same height. Above 1 gains
    /// energy and is clamped, because a solver that can add energy will
    /// find a way to add it forever.
    pub restitution: f32,
    /// Coulomb friction coefficient. 0 slides freely; ~0.6 is wood on
    /// wood; above ~1.5 is effectively glued.
    pub friction: f32,
}

impl Default for Material3D {
    fn default() -> Self {
        // A slightly bouncy, ordinarily grippy surface. Chosen so a
        // dropped box settles rather than either sticking on impact or
        // bouncing around the room.
        Self { restitution: 0.2, friction: 0.5 }
    }
}

/// A body the solver can move.
///
/// Sits beside [`crate::components::Transform3D`] and
/// [`crate::components::Velocity3D`]: the transform is where it is, the
/// velocity is how fast, and this is everything needed to work out how it
/// responds to being hit.
///
/// # Inverse mass, not mass
///
/// Both mass and inertia are stored inverted. Every use in the solver is
/// a division by them, and a static body is the `0.0` case rather than an
/// infinity that has to be special-cased at each site.
#[cfg_attr(feature = "persist", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct RigidBody {
    pub kind: BodyKind,
    /// 1/mass. Zero for static and kinematic bodies.
    pub inv_mass: f32,
    /// Inverse inertia tensor in **body space**. Rotated into world
    /// space per solve, because a tumbling body's world-space inertia
    /// changes every frame while its body-space inertia never does.
    pub inv_inertia: Mat3,
    pub material: Material3D,
    /// Per-tick multipliers, matching the 2D integrator's convention.
    pub linear_damping: f64,
    pub angular_damping: f64,
    /// Whether this body is currently asleep and skipped by the solver.
    ///
    /// See [`crate::physics3d::update_sleep`] — a scene of settled objects
    /// should cost nothing, and without sleeping a stack of crates jitters
    /// forever at the solver's error floor.
    pub sleeping: bool,
    /// Seconds this body has been below the sleep threshold.
    pub time_below_threshold: f32,
}

impl Default for RigidBody {
    fn default() -> Self {
        Self {
            kind: BodyKind::Dynamic,
            inv_mass: 1.0,
            inv_inertia: Mat3::IDENTITY,
            material: Material3D::default(),
            linear_damping: 1.0,
            angular_damping: 1.0,
            sleeping: false,
            time_below_threshold: 0.0,
        }
    }
}

impl RigidBody {
    /// A dynamic body of the given mass, with a solid-sphere inertia.
    ///
    /// The sphere tensor is the forgiving default: it is isotropic, so a
    /// body whose real shape is unknown tumbles plausibly rather than
    /// preferring an axis it should not.
    pub fn sphere(mass: f32, radius: f32) -> Self {
        let m = mass.max(MIN_MASS);
        let r = radius.abs().max(MIN_EXTENT);
        // I = 2/5 m r² about every axis.
        let i = (0.4 * m * r * r).max(f32::MIN_POSITIVE);
        Self {
            inv_mass: 1.0 / m,
            inv_inertia: Mat3::from_diagonal(glam::Vec3::splat(1.0 / i)),
            ..Default::default()
        }
    }

    /// A dynamic body of the given mass shaped like a box.
    ///
    /// `half_extents` matches [`crate::components::Collider3D`], so the
    /// two are built from the same numbers.
    pub fn box3d(mass: f32, half_extents: [f32; 3]) -> Self {
        let m = mass.max(MIN_MASS);
        // Clamped well above `f32::MIN_POSITIVE`, which is *not* enough
        // here: the tensor squares these and multiplies by m/12, so a
        // denormal extent underflows the product to zero and `1/0` puts
        // an infinity in the tensor. A millimetre is the smallest extent
        // any of this is meaningful at.
        let (x, y, z) = (
            (half_extents[0].abs() * 2.0).max(MIN_EXTENT),
            (half_extents[1].abs() * 2.0).max(MIN_EXTENT),
            (half_extents[2].abs() * 2.0).max(MIN_EXTENT),
        );
        // I = m/12 * (a² + b²) per axis, over the *full* extents.
        let k = m / 12.0;
        let ix = (k * (y * y + z * z)).max(f32::MIN_POSITIVE);
        let iy = (k * (x * x + z * z)).max(f32::MIN_POSITIVE);
        let iz = (k * (x * x + y * y)).max(f32::MIN_POSITIVE);
        Self {
            inv_mass: 1.0 / m,
            inv_inertia: Mat3::from_diagonal(glam::Vec3::new(
                1.0 / ix,
                1.0 / iy,
                1.0 / iz,
            )),
            ..Default::default()
        }
    }

    /// Immovable level geometry.
    pub fn static_body() -> Self {
        Self {
            kind: BodyKind::Static,
            inv_mass: 0.0,
            inv_inertia: Mat3::ZERO,
            ..Default::default()
        }
    }

    /// Moved by its transform, unmoved by collisions.
    pub fn kinematic() -> Self {
        Self {
            kind: BodyKind::Kinematic,
            inv_mass: 0.0,
            inv_inertia: Mat3::ZERO,
            ..Default::default()
        }
    }

    pub fn with_material(mut self, material: Material3D) -> Self {
        self.material = material;
        self
    }

    /// Whether the solver should move this body right now.
    ///
    /// Both conditions matter: a static body never moves, and a sleeping
    /// dynamic one is skipped until something wakes it.
    pub fn is_movable(&self) -> bool {
        self.kind.is_dynamic() && !self.sleeping
    }

    /// Wake a sleeping body and reset its timer.
    ///
    /// Called whenever something touches it. A body that is hit while
    /// asleep and *stays* asleep is the classic "bullet passes through
    /// the crate" bug.
    pub fn wake(&mut self) {
        self.sleeping = false;
        self.time_below_threshold = 0.0;
    }

    /// The inverse inertia tensor in world space for a given orientation.
    ///
    /// `R * I⁻¹ * Rᵀ`. Recomputed per solve rather than stored, because
    /// it is only valid for the orientation it was built from.
    pub fn world_inv_inertia(&self, rot: Quat) -> Mat3 {
        let r = Mat3::from_quat(rot.normalize());
        r * self.inv_inertia * r.transpose()
    }

    /// Restitution for a pair. The **maximum** of the two.
    ///
    /// Max rather than a product or average so a bouncy ball stays bouncy
    /// against a dead floor — which is what someone setting
    /// `restitution: 0.9` on the ball expects. A product would make every
    /// surface it touches cancel it out.
    pub fn combined_restitution(a: &Self, b: &Self) -> f32 {
        a.material.restitution.max(b.material.restitution).clamp(0.0, 1.0)
    }

    /// Friction for a pair. The geometric mean of the two.
    ///
    /// The conventional choice: ice on ice is slippery, ice on rubber is
    /// somewhere between, and either surface being frictionless makes the
    /// pair frictionless — which matches intuition better than an average.
    pub fn combined_friction(a: &Self, b: &Self) -> f32 {
        (a.material.friction.max(0.0) * b.material.friction.max(0.0)).sqrt()
    }
}

/// Apply an impulse at a body's centre of mass.
///
/// Changes linear velocity only, leaving spin untouched — which is what
/// "at the centre of mass" means. For an off-centre hit use
/// [`apply_impulse_at`].
pub fn apply_impulse(body: &RigidBody, vel: &mut crate::components::Velocity3D, impulse: DVec3) {
    if !body.kind.is_dynamic() {
        return;
    }
    vel.linear += impulse * body.inv_mass as f64;
}

/// Apply an impulse at a world-space offset from the centre of mass.
///
/// Off-centre impulses impart spin, which is the whole reason a body has
/// an inertia tensor. `rel` is the contact point relative to the centre.
pub fn apply_impulse_at(
    body: &RigidBody,
    rot: Quat,
    vel: &mut crate::components::Velocity3D,
    impulse: DVec3,
    rel: DVec3,
) {
    if !body.kind.is_dynamic() {
        return;
    }
    vel.linear += impulse * body.inv_mass as f64;
    let torque = rel.cross(impulse);
    let world_inv_i = body.world_inv_inertia(rot);
    vel.angular += (world_inv_i * torque.as_vec3()).as_dvec3();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_static_body_has_zero_inverse_mass_so_it_never_accelerates() {
        let b = RigidBody::static_body();
        assert_eq!(b.inv_mass, 0.0);
        assert_eq!(b.inv_inertia, Mat3::ZERO);
        assert!(!b.kind.is_dynamic());
    }

    /// The whole point of storing the inverse: a static body is the zero
    /// case, so `impulse * inv_mass` is naturally a no-op rather than
    /// needing a branch at every use.
    #[test]
    fn an_impulse_does_not_move_a_static_body() {
        let b = RigidBody::static_body();
        let mut v = crate::components::Velocity3D::default();
        apply_impulse(&b, &mut v, DVec3::new(100.0, 0.0, 0.0));
        assert_eq!(v.linear, DVec3::ZERO);
    }

    #[test]
    fn a_centre_impulse_adds_no_spin() {
        let b = RigidBody::sphere(2.0, 1.0);
        let mut v = crate::components::Velocity3D::default();
        apply_impulse(&b, &mut v, DVec3::new(10.0, 0.0, 0.0));
        assert!((v.linear.x - 5.0).abs() < 1e-9, "10 N·s on 2 kg is 5 m/s");
        assert_eq!(v.angular, DVec3::ZERO);
    }

    /// An off-centre hit must spin the body. This is what separates a
    /// rigid body from a point mass, and getting the cross product
    /// backwards spins everything the wrong way.
    #[test]
    fn an_off_centre_impulse_adds_spin_about_the_expected_axis() {
        let b = RigidBody::sphere(1.0, 1.0);
        let mut v = crate::components::Velocity3D::default();
        // Push +X at a point +Z of centre: torque = r × F = ẑ × x̂ = ŷ.
        apply_impulse_at(
            &b,
            Quat::IDENTITY,
            &mut v,
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(0.0, 0.0, 1.0),
        );
        assert!(v.angular.y > 0.0, "expected +Y spin, got {:?}", v.angular);
        assert!(v.angular.x.abs() < 1e-9 && v.angular.z.abs() < 1e-9);
    }

    /// A long thin box must be easier to spin about its long axis than
    /// across it. Getting the box tensor's axes transposed is easy and
    /// makes objects tumble subtly wrong rather than obviously broken.
    #[test]
    fn a_long_box_spins_most_easily_about_its_long_axis() {
        // Long in x.
        let b = RigidBody::box3d(1.0, [4.0, 0.5, 0.5]);
        let inv = b.inv_inertia.to_cols_array_2d();
        // Larger inverse inertia = easier to spin.
        assert!(
            inv[0][0] > inv[1][1] && inv[0][0] > inv[2][2],
            "spinning about the long (x) axis should be easiest; got {inv:?}",
        );
    }

    /// Max, not product: a bouncy ball must stay bouncy against a dead
    /// floor, which is what setting a high restitution on the ball means.
    #[test]
    fn restitution_combines_as_the_max_so_a_bouncy_ball_stays_bouncy() {
        let ball = RigidBody::default()
            .with_material(Material3D { restitution: 0.9, friction: 0.5 });
        let floor = RigidBody::static_body()
            .with_material(Material3D { restitution: 0.0, friction: 0.5 });
        assert!((RigidBody::combined_restitution(&ball, &floor) - 0.9).abs() < 1e-6);
    }

    /// Restitution above 1 adds energy every bounce, and a solver that
    /// can add energy will find a way to add it forever.
    #[test]
    fn restitution_is_clamped_so_a_bounce_cannot_gain_energy() {
        let a = RigidBody::default()
            .with_material(Material3D { restitution: 5.0, friction: 0.0 });
        assert_eq!(RigidBody::combined_restitution(&a, &a), 1.0);
    }

    /// Either surface being frictionless makes the pair frictionless,
    /// which matches intuition better than an average would.
    #[test]
    fn a_frictionless_surface_makes_the_whole_pair_frictionless() {
        let ice = RigidBody::default()
            .with_material(Material3D { restitution: 0.0, friction: 0.0 });
        let rubber = RigidBody::default()
            .with_material(Material3D { restitution: 0.0, friction: 1.2 });
        assert_eq!(RigidBody::combined_friction(&ice, &rubber), 0.0);
    }

    /// World inertia must follow the body's orientation. A tumbling
    /// body's resistance to spin changes as it turns; using the
    /// body-space tensor directly is a bug that only shows on rotated
    /// objects.
    #[test]
    fn world_inertia_follows_the_bodys_orientation() {
        let b = RigidBody::box3d(1.0, [4.0, 0.5, 0.5]);
        let upright = b.world_inv_inertia(Quat::IDENTITY);
        // Turn the long axis from x onto y.
        let turned = b.world_inv_inertia(Quat::from_rotation_z(
            std::f32::consts::FRAC_PI_2,
        ));
        let u = upright.to_cols_array_2d();
        let t = turned.to_cols_array_2d();
        assert!(
            (u[0][0] - t[1][1]).abs() < 1e-4,
            "after a quarter turn about z, the easy axis should have moved \
             from x to y; got {u:?} then {t:?}",
        );
    }

    #[test]
    fn a_sleeping_body_is_not_movable_until_woken() {
        let mut b = RigidBody { sleeping: true, ..Default::default() };
        assert!(!b.is_movable());
        b.wake();
        assert!(b.is_movable());
        assert_eq!(b.time_below_threshold, 0.0);
    }

    /// Zero or negative mass would make `1/m` infinite or negative, and a
    /// negative inverse mass accelerates a body *toward* what hits it.
    ///
    /// The inertia tensor is the harder half and caught a real bug:
    /// clamping extents at `f32::MIN_POSITIVE` is not enough, because the
    /// tensor squares them and scales by m/12, so the product still
    /// underflows to zero and `1/0` puts an infinity in the tensor. An
    /// infinite inverse inertia means any off-centre impulse spins the
    /// body at an unrepresentable rate, which poisons its pose and every
    /// contact it takes part in thereafter.
    #[test]
    fn a_nonsense_mass_or_size_does_not_produce_an_infinite_inverse() {
        let degenerate = [
            RigidBody::sphere(0.0, 1.0),
            RigidBody::sphere(-5.0, 1.0),
            RigidBody::sphere(1.0, 0.0),
            RigidBody::sphere(1.0, -2.0),
            RigidBody::box3d(0.0, [0.0, 0.0, 0.0]),
            RigidBody::box3d(-1.0, [1.0, 1.0, 1.0]),
            RigidBody::box3d(1.0, [0.0, 0.0, 0.0]),
            RigidBody::box3d(1.0, [-1.0, -1.0, -1.0]),
        ];
        for (i, b) in degenerate.iter().enumerate() {
            assert!(
                b.inv_mass.is_finite() && b.inv_mass > 0.0,
                "case {i}: inverse mass was {}",
                b.inv_mass,
            );
            assert!(
                b.inv_inertia.to_cols_array().iter().all(|v| v.is_finite()),
                "case {i}: inertia tensor was {:?}",
                b.inv_inertia,
            );
        }
    }

    /// A negative extent is a caller error, not a mirrored box: it must
    /// come out the same as its positive twin rather than producing a
    /// negative inertia, which would spin the body the wrong way.
    #[test]
    fn negative_extents_behave_like_their_positive_twins() {
        let a = RigidBody::box3d(2.0, [1.0, 2.0, 3.0]);
        let b = RigidBody::box3d(2.0, [-1.0, -2.0, -3.0]);
        assert_eq!(a.inv_inertia, b.inv_inertia);
    }
}
