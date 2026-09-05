//! Seeded value noise and fBm.
//!
//! Pure functions of `(position, seed)` — no RNG state is threaded through — so
//! a feature placed by noise stays put as long as the seed does, regardless of
//! the order in which callers sample it. That property is what lets terrain be
//! evaluated lazily per-tile instead of baked into a grid up front.

use glam::Vec2;

/// Hash a lattice point to a pseudo-random f32 in [0, 1).
fn hash2(ix: i32, iy: i32, seed: u64) -> f32 {
    let mut h = seed
        ^ (ix as i64 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (iy as i64 as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    h ^= h >> 33;
    (h >> 40) as f32 / (1u32 << 24) as f32
}

/// Ken Perlin's quintic ease curve. Smoother than `smoothstep` at the lattice
/// points (its second derivative is also zero there), which keeps fBm free of
/// the faint grid-aligned creasing you get from a cubic fade.
fn smootherstep(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Value noise in roughly [0, 1], continuous, one octave.
pub fn value_noise(p: Vec2, seed: u64) -> f32 {
    let ix = p.x.floor() as i32;
    let iy = p.y.floor() as i32;
    let fx = p.x - ix as f32;
    let fy = p.y - iy as f32;

    let v00 = hash2(ix, iy, seed);
    let v10 = hash2(ix + 1, iy, seed);
    let v01 = hash2(ix, iy + 1, seed);
    let v11 = hash2(ix + 1, iy + 1, seed);

    let ux = smootherstep(fx);
    let uy = smootherstep(fy);
    let a = v00 + (v10 - v00) * ux;
    let b = v01 + (v11 - v01) * ux;
    a + (b - a) * uy
}

/// Fractal Brownian motion — the sum of `octaves` value-noise layers, each at
/// double the frequency and half the amplitude of the last. Returns a value in
/// roughly [-1, 1], centered so it warps boundaries in both directions.
pub fn fbm(p: Vec2, octaves: u32, seed: u64) -> f32 {
    let mut sum = 0.0;
    let mut amp = 0.5;
    let mut freq = 1.0;
    let mut norm = 0.0;
    for o in 0..octaves {
        sum += amp * value_noise(p * freq, seed ^ (o as u64).wrapping_mul(0x1234567));
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    (sum / norm) * 2.0 - 1.0
}

/// Sample `f` at a position displaced by its own noise — "domain warping".
/// Straight fBm gives blobs that read as obviously procedural; warping the
/// input first bends those blobs into the wandering, non-radial shapes that
/// coastlines and relief actually have.
///
/// `strength` is in world units: how far a point can be displaced before
/// sampling.
pub fn warped(p: Vec2, strength: f32, seed: u64, f: impl Fn(Vec2) -> f32) -> f32 {
    let w = Vec2::new(
        fbm(p * 0.00022 + Vec2::new(31.4, 0.0), 2, seed ^ 0x1111),
        fbm(p * 0.00022 + Vec2::new(0.0, 27.2), 2, seed ^ 0x2222),
    ) * strength;
    f(p + w)
}

/// Ridged noise in [0, 1]: `1 - |fbm|`, which turns fBm's zero-crossings into
/// sharp crests. Multiply into a mountain profile for craggy peaks rather than
/// smooth domes.
pub fn ridged(p: Vec2, octaves: u32, seed: u64) -> f32 {
    1.0 - fbm(p, octaves, seed).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_noise_is_deterministic() {
        let p = Vec2::new(12.5, -3.25);
        assert_eq!(value_noise(p, 42), value_noise(p, 42));
        assert_ne!(value_noise(p, 42), value_noise(p, 43));
    }

    #[test]
    fn value_noise_stays_in_unit_range() {
        for i in 0..500 {
            let p = Vec2::new(i as f32 * 0.37, i as f32 * -0.11);
            let v = value_noise(p, 7);
            assert!((0.0..=1.0).contains(&v), "value_noise out of range: {v}");
        }
    }

    #[test]
    fn value_noise_is_continuous_across_a_lattice_line() {
        // Either side of the integer boundary at x = 4 should agree closely.
        let a = value_noise(Vec2::new(4.0 - 1e-4, 0.3), 9);
        let b = value_noise(Vec2::new(4.0 + 1e-4, 0.3), 9);
        assert!((a - b).abs() < 1e-3, "discontinuity at lattice line: {a} vs {b}");
    }

    #[test]
    fn fbm_is_centered_and_bounded() {
        let mut sum = 0.0;
        let n = 2000;
        for i in 0..n {
            let p = Vec2::new(i as f32 * 0.13, i as f32 * 0.29);
            let v = fbm(p, 4, 3);
            assert!((-1.0..=1.0).contains(&v), "fbm out of range: {v}");
            sum += v;
        }
        // Centered on zero, so a long average should be near it.
        assert!((sum / n as f32).abs() < 0.15);
    }

    #[test]
    fn ridged_is_in_unit_range() {
        for i in 0..300 {
            let p = Vec2::new(i as f32 * 0.21, 5.0);
            let v = ridged(p, 4, 11);
            assert!((0.0..=1.0).contains(&v), "ridged out of range: {v}");
        }
    }

    #[test]
    fn warped_displaces_the_sample() {
        // A ramp in x: warping must move the sample point, so the result should
        // differ from the unwarped value.
        let p = Vec2::new(1000.0, 2000.0);
        let f = |q: Vec2| q.x;
        assert_ne!(warped(p, 900.0, 5, f), f(p));
        // Zero strength is the identity.
        assert_eq!(warped(p, 0.0, 5, f), f(p));
    }
}
