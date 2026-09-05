//! Deterministic, seedable pseudo-random number generator.
//!
//! Linear congruential generator using Knuth's MMIX constants
//! (`a = 6364136223846793005`, `c = 1442695040888963407`). It's the
//! same LCG the galaxy procgen uses and the same one the void_sim
//! `rand_f32` shim uses under the hood; extracted so any consumer that
//! needs a small, cheap, reproducible RNG (particles, cosmetic FX, unit
//! tests) can construct one from an explicit seed.
//!
//! Not cryptographically secure. Not suitable for anything an adversary
//! could exploit. Fine for game state that must reproduce given the
//! same seed.

pub struct Lcg(pub u64);

impl Lcg {
    /// Construct with an explicit seed. Same seed → same sequence.
    pub fn new(seed: u64) -> Self { Self(seed) }

    /// Advance and return the next raw 64-bit state.
    // Not `Iterator::next`: this stream is infinite and returns `u64`, not
    // `Option<u64>`. Implementing Iterator would force an `Option` wrapper
    // on every call in the hot particle//spawn paths for no benefit.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> u64 {
        self.0 = self.0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    /// Uniform `[0, 1)` as f64. Uses the top 53 bits of state (the full
    /// f64 mantissa) so the distribution is exactly representable.
    pub fn f64_01(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform `[0, 1)` as f32. Uses the top 31 bits so the divisor
    /// (`2^31`) is exact in f32 and no bucket is dropped.
    pub fn f32_01(&mut self) -> f32 {
        (self.next() >> 33) as f32 / (1u32 << 31) as f32
    }

    /// Uniform `[lo, hi)` as f64.
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.f64_01() * (hi - lo)
    }

    /// Uniform `[lo, hi)` as f32.
    pub fn rangef(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.f32_01() * (hi - lo)
    }
}
