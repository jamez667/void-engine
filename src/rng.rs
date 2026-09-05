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

/// PCG32 — a small, seedable generator with *independent streams*.
///
/// [`Lcg`] is fine when one sequence is all you need. Worldgen usually is not
/// like that: it runs as a pipeline of stages, and if every stage draws from
/// one shared stream then tweaking an early stage reshuffles everything
/// downstream — change the number of rivers and the forests move. [`fork`]
/// gives each stage its own statistically independent sequence, so a stage can
/// be re-tuned without disturbing its neighbours.
///
/// The output function (an xorshift plus a random rotation) also passes
/// statistical tests a bare LCG fails, at essentially the same cost.
///
/// [`fork`]: Pcg32::fork
#[derive(Clone)]
pub struct Pcg32 {
    state: u64,
    inc: u64,
}

const PCG_MUL: u64 = 6364136223846793005;

impl Pcg32 {
    /// Seed with a value and a stream selector. Different `stream` values give
    /// statistically independent sequences from the same `seed`.
    pub fn seed(seed: u64, stream: u64) -> Self {
        let mut r = Pcg32 { state: 0, inc: (stream << 1) | 1 };
        r.next_u32();
        r.state = r.state.wrapping_add(seed);
        r.next_u32();
        r
    }

    /// Derive an independent sub-stream (e.g. for one generation stage) from
    /// this RNG's current state, advancing self by a single step.
    pub fn fork(&mut self, stream: u64) -> Pcg32 {
        Pcg32::seed(self.next_u64(), stream)
    }

    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(PCG_MUL).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    pub fn next_u64(&mut self) -> u64 {
        (self.next_u32() as u64) << 32 | self.next_u32() as u64
    }

    /// Uniform f32 in `[0, 1)`, with 24 bits of mantissa precision.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// Uniform f32 in `[lo, hi)`.
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }

    /// Uniform integer in `[lo, hi]`, inclusive.
    pub fn range_i(&mut self, lo: i32, hi: i32) -> i32 {
        if hi <= lo {
            return lo;
        }
        let span = (hi - lo + 1) as u32;
        lo + (self.next_u32() % span) as i32
    }

    /// True with probability `p`.
    pub fn chance(&mut self, p: f32) -> bool {
        self.next_f32() < p
    }

    /// Pick a random element from a non-empty slice.
    ///
    /// # Panics
    /// If `items` is empty.
    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        let i = (self.next_u32() as usize) % items.len();
        &items[i]
    }
}

#[cfg(test)]
mod pcg_tests {
    use super::Pcg32;

    #[test]
    fn same_seed_same_sequence() {
        let mut a = Pcg32::seed(99, 1);
        let mut b = Pcg32::seed(99, 1);
        for _ in 0..64 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn streams_are_independent() {
        let mut a = Pcg32::seed(99, 1);
        let mut b = Pcg32::seed(99, 2);
        // Same seed, different stream: the sequences must not coincide.
        let differ = (0..32).filter(|_| a.next_u32() != b.next_u32()).count();
        assert!(differ > 28, "streams too correlated: only {differ}/32 differed");
    }

    #[test]
    fn fork_is_reproducible_and_distinct() {
        let mut parent = Pcg32::seed(7, 0);
        let mut again = Pcg32::seed(7, 0);
        let mut f1 = parent.fork(1);
        let mut f2 = again.fork(1);
        assert_eq!(f1.next_u32(), f2.next_u32());

        // Forking a different stream off the same parent state diverges.
        let mut p = Pcg32::seed(7, 0);
        let mut g1 = p.fork(1);
        let mut p2 = Pcg32::seed(7, 0);
        let mut g2 = p2.fork(2);
        assert_ne!(g1.next_u32(), g2.next_u32());
    }

    #[test]
    fn f32_is_in_unit_range() {
        let mut r = Pcg32::seed(1, 1);
        for _ in 0..5000 {
            let v = r.next_f32();
            assert!((0.0..1.0).contains(&v), "next_f32 out of range: {v}");
        }
    }

    #[test]
    fn range_i_is_inclusive_and_covers_its_span() {
        let mut r = Pcg32::seed(5, 3);
        let mut seen = [false; 4];
        for _ in 0..500 {
            let v = r.range_i(2, 5);
            assert!((2..=5).contains(&v), "range_i out of bounds: {v}");
            seen[(v - 2) as usize] = true;
        }
        assert!(seen.iter().all(|&s| s), "range_i never produced some values");
        // Degenerate span returns lo.
        assert_eq!(r.range_i(9, 9), 9);
        assert_eq!(r.range_i(9, 1), 9);
    }

    #[test]
    fn chance_respects_its_probability() {
        let mut r = Pcg32::seed(11, 4);
        assert!((0..100).all(|_| r.chance(1.0)));
        assert!((0..100).all(|_| !r.chance(0.0)));
    }
}
