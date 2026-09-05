//! Heightfield primitives: the small pure functions terrain generators are
//! built out of.
//!
//! Nothing here knows what a world looks like. A generator supplies its own
//! elevation recipe — where the mountains go, how the coast is shaped — and
//! reaches for these to shape it: ramps to blend one region into another,
//! point-to-segment distance for anything defined by a polyline, and the two
//! erosion-ish operators (valley carving and delta fans) that make water and
//! land agree with each other.

use glam::Vec2;

/// Smooth Hermite ramp from `edge0` to `edge1`: 0 below `edge0`, 1 above
/// `edge1`, with an ease in between.
///
/// Works with `edge0 > edge1` too, which gives a descending ramp — useful for
/// masks that fade *out* with distance, like a landmass falling away to sea.
#[inline]
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Squared distance from point `p` to segment `ab`.
///
/// Squared, because callers overwhelmingly compare against a squared radius;
/// taking the root at every segment of every polyline is pure waste.
#[inline]
pub fn point_seg_dist2(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let len2 = ab.length_squared();
    if len2 < 1e-6 {
        return (p - a).length_squared();
    }
    let t = ((p - a).dot(ab) / len2).clamp(0.0, 1.0);
    (p - (a + ab * t)).length_squared()
}

/// Distance from point `p` to segment `ab`.
#[inline]
pub fn point_seg_dist(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    point_seg_dist2(p, a, b).sqrt()
}

/// Lower `land` toward a valley floor near a watercourse: a gorge through
/// highlands, a shallow dip across plains.
///
/// `dist` is the distance to the nearest channel and `corridor` how wide the
/// valley is. The floor is clamped so carving never raises ground that was
/// already lower, and never digs below `floor` — which matters, because a
/// valley cut below sea level would read as a fake inland ocean.
pub fn carve_valley(land: f32, dist: f32, corridor: f32, floor: f32) -> f32 {
    if dist >= corridor || corridor <= 0.0 {
        return land;
    }
    let t = 1.0 - dist / corridor;
    let carve = t * t * (3.0 - 2.0 * t);
    let target = floor.min(land);
    land + (target - land) * carve
}

/// Flatten terrain toward `level` within `radius` of `center` — the alluvial
/// fan a river builds where it meets the sea, and incidentally the generic
/// "flatten a disc of ground" operator.
///
/// The falloff is quadratic rather than smooth so the fan stays flat across
/// most of its area and steepens only at the rim, which is what a real delta
/// looks like and what makes it useful as buildable ground.
pub fn apply_delta(elev: f32, dist: f32, radius: f32, level: f32) -> f32 {
    if dist >= radius || radius <= 0.0 {
        return elev;
    }
    let t = 1.0 - dist / radius;
    elev + (level - elev) * t * t
}

/// A radial mask that is 1 inside `inner` and 0 outside `outer`, with a smooth
/// coastline-shaped falloff between. The building block of an island: multiply
/// land features by it and they fade into the sea instead of ending at a cliff.
#[inline]
pub fn radial_mask(dist: f32, inner: f32, outer: f32) -> f32 {
    smoothstep(outer, inner, dist)
}

/// Numerical gradient of a heightfield at `p`, by central differences over
/// `eps`. Points uphill; negate it for steepest descent.
pub fn gradient(p: Vec2, eps: f32, height: impl Fn(Vec2) -> f32) -> Vec2 {
    let e = eps.max(1e-4);
    let dx = height(p + Vec2::new(e, 0.0)) - height(p - Vec2::new(e, 0.0));
    let dy = height(p + Vec2::new(0.0, e)) - height(p - Vec2::new(0.0, e));
    Vec2::new(dx, dy) / (2.0 * e)
}

/// Slope magnitude at `p` — the steepness of the field, in height units per
/// world unit. Useful for deciding what is buildable, walkable, or bare rock.
pub fn slope(p: Vec2, eps: f32, height: impl Fn(Vec2) -> f32) -> f32 {
    gradient(p, eps, height).length()
}

/// Lambertian hillshade in [0, 1] for a light coming from `light_dir`
/// (a direction in the XY plane, normalized internally).
///
/// `relief` exaggerates the terrain's vertical scale before shading; a real
/// landscape's slopes are far too gentle to read as shape at map scale, so
/// this is what makes hills visible at all.
pub fn hillshade(p: Vec2, eps: f32, relief: f32, light_dir: Vec2, height: impl Fn(Vec2) -> f32) -> f32 {
    let g = gradient(p, eps, height) * relief;
    // Surface normal of the heightfield: (-dz/dx, -dz/dy, 1), normalized.
    let n = glam::Vec3::new(-g.x, -g.y, 1.0).normalize();
    let l = light_dir.normalize_or_zero();
    let l = glam::Vec3::new(l.x, l.y, 0.6).normalize();
    n.dot(l).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoothstep_clamps_and_eases() {
        assert_eq!(smoothstep(0.0, 1.0, -5.0), 0.0);
        assert_eq!(smoothstep(0.0, 1.0, 5.0), 1.0);
        assert!((smoothstep(0.0, 1.0, 0.5) - 0.5).abs() < 1e-6);
        // Descending edges give a falling ramp.
        assert_eq!(smoothstep(1.0, 0.0, -5.0), 1.0);
        assert_eq!(smoothstep(1.0, 0.0, 5.0), 0.0);
    }

    #[test]
    fn point_seg_dist_handles_ends_and_middle() {
        let (a, b) = (Vec2::ZERO, Vec2::new(10.0, 0.0));
        // Perpendicular from the middle.
        assert!((point_seg_dist(Vec2::new(5.0, 3.0), a, b) - 3.0).abs() < 1e-5);
        // Past an endpoint clamps to that endpoint, not the infinite line.
        assert!((point_seg_dist(Vec2::new(-4.0, 0.0), a, b) - 4.0).abs() < 1e-5);
        assert!((point_seg_dist(Vec2::new(14.0, 0.0), a, b) - 4.0).abs() < 1e-5);
        // Degenerate segment behaves as a point.
        assert!((point_seg_dist(Vec2::new(3.0, 4.0), a, a) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn carve_valley_only_lowers_and_respects_the_floor() {
        // Far outside the corridor: untouched.
        assert_eq!(carve_valley(500.0, 900.0, 100.0, 0.0), 500.0);
        // At the channel: pulled fully to the floor.
        assert!((carve_valley(500.0, 0.0, 100.0, 3.0) - 3.0).abs() < 1e-3);
        // Partway: between the two, and strictly lower than the land.
        let mid = carve_valley(500.0, 50.0, 100.0, 3.0);
        assert!(mid > 3.0 && mid < 500.0, "mid was {mid}");
        // Ground already below the floor is never raised.
        assert!(carve_valley(1.0, 0.0, 100.0, 3.0) <= 1.0);
    }

    #[test]
    fn apply_delta_flattens_toward_its_level() {
        assert_eq!(apply_delta(300.0, 200.0, 100.0, 2.0), 300.0, "outside is untouched");
        assert!((apply_delta(300.0, 0.0, 100.0, 2.0) - 2.0).abs() < 1e-3, "center reaches level");
        let edge = apply_delta(300.0, 90.0, 100.0, 2.0);
        assert!(edge > 250.0, "rim should barely change: {edge}");
        assert_eq!(apply_delta(300.0, 0.0, 0.0, 2.0), 300.0, "zero radius is a no-op");
    }

    #[test]
    fn radial_mask_is_one_inside_and_zero_outside() {
        assert_eq!(radial_mask(0.0, 100.0, 200.0), 1.0);
        assert_eq!(radial_mask(500.0, 100.0, 200.0), 0.0);
        let mid = radial_mask(150.0, 100.0, 200.0);
        assert!(mid > 0.0 && mid < 1.0, "mid was {mid}");
    }

    #[test]
    fn gradient_points_uphill() {
        // Height rises with x, so the gradient is +x.
        let h = |p: Vec2| p.x * 2.0;
        let g = gradient(Vec2::new(5.0, 5.0), 0.01, h);
        assert!((g.x - 2.0).abs() < 1e-3, "g.x = {}", g.x);
        assert!(g.y.abs() < 1e-3);
        assert!((slope(Vec2::new(5.0, 5.0), 0.01, h) - 2.0).abs() < 1e-3);
    }

    #[test]
    fn flat_ground_has_no_slope() {
        let h = |_p: Vec2| 42.0;
        assert!(slope(Vec2::new(1.0, 2.0), 0.01, h) < 1e-6);
    }

    #[test]
    fn hillshade_is_brighter_facing_the_light() {
        // A slope descending toward +x: its normal tilts toward +x, so a light
        // from +x lights it more than a light from -x.
        let h = |p: Vec2| -p.x;
        let p = Vec2::new(10.0, 0.0);
        let lit = hillshade(p, 0.01, 1.0, Vec2::X, h);
        let away = hillshade(p, 0.01, 1.0, -Vec2::X, h);
        assert!(lit > away, "lit {lit} should exceed away {away}");
        assert!((0.0..=1.0).contains(&lit));
        assert!((0.0..=1.0).contains(&away));
    }
}
