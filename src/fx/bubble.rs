//! Warp-bubble "pop" — a world-space travel effect for any 2D game with
//! a teleport / jump / dive mechanic.
//!
//! The read we're after is *reality closing over the ship*, not an
//! explosion. Three phases:
//!
//!   1. **Inflate** — a translucent membrane swells around the ship.
//!      Its surface *thins* as it grows (alpha falls with radius) so it
//!      reads as a soap bubble being stretched past its limit, and a
//!      thin ring of motes drifts *inward* toward the ship, feeding it.
//!   2. **Snap** — one frame. The membrane vanishes; a burst of motes
//!      fires outward from exactly where the membrane was standing (the
//!      *rim*, not the centre — that is what makes it read as a surface
//!      letting go rather than a core detonating). The caller hides the
//!      ship on this frame, so the pop visually *causes* the departure.
//!   3. **Afterglow** — a fast-fading residual ring + glow. No membrane
//!      survives; the particles carry it out.
//!
//! Arrival is the same curve run backwards (`Direction::Arrive`): a
//! bright point swells, bursts open, and the ship is there.
//!
//! Rendering deliberately uses only flat-coloured geometry + the shared
//! particle pool — the renderer has no shader/material path, so the
//! "translucent membrane" is built from concentric alpha-graded circle
//! outlines rather than a gradient fill.

use std::f32::consts::TAU;
use glam::{DVec2, Vec2};

use crate::World;
use crate::renderer::batch::Batch;

/// Which way the effect runs. `Depart` inflates then pops; `Arrive`
/// plays the identical curve in reverse so a bright point swells and
/// bursts open.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Direction { Depart, Arrive }

/// Tuning knobs for one bubble. Bundled into a struct rather than
/// passed positionally — there are far too many for a sane argument
/// list, and games want to name these as presets.
#[derive(Clone, Copy, Debug)]
pub struct BubbleParams {
    /// Seconds the membrane spends inflating before the snap.
    pub inflate_secs: f32,
    /// Seconds the afterglow ring lingers after the snap.
    pub afterglow_secs: f32,
    /// Peak membrane radius in world units, reached at the snap.
    pub radius: f32,
    /// Particles fired outward from the rim on the snap frame.
    pub burst_count: u32,
    /// Motes drifting inward along the membrane during inflate.
    pub feed_count: u32,
    /// Outward speed range of the snap burst, in world units/sec.
    pub burst_speed: (f64, f64),
    /// Membrane tint, linear RGB. Alpha is derived from the phase.
    pub color: [f32; 3],
    /// Hot inner colour for the snap flash particles.
    pub hot_color: [f32; 3],
    /// Peak light intensity at the snap. 0 disables the flash.
    pub flash_intensity: f32,
    /// Light radius multiplier relative to `radius`.
    pub flash_radius_mul: f32,
}

impl BubbleParams {
    /// Total wall-clock life of the effect.
    pub fn total_secs(&self) -> f32 { self.inflate_secs + self.afterglow_secs }
}

/// A live bubble instance. Drive with [`BubblePop::update`], draw with
/// [`BubblePop::draw`], and ask [`BubblePop::ship_hidden`] whether the
/// travelling object should be suppressed this frame.
#[derive(Clone, Debug)]
pub struct BubblePop {
    pub params: BubbleParams,
    pub dir: Direction,
    /// World position the bubble is anchored to.
    pub pos: DVec2,
    /// Seconds since spawn.
    pub elapsed: f32,
    /// False until the snap frame has fired its burst.
    snapped: bool,
}

impl BubblePop {
    pub fn new(pos: DVec2, dir: Direction, params: BubbleParams) -> Self {
        Self { params, dir, pos, elapsed: 0.0, snapped: false }
    }

    /// Normalised progress across the whole effect, 0..=1.
    pub fn progress(&self) -> f32 {
        (self.elapsed / self.params.total_secs().max(1e-4)).clamp(0.0, 1.0)
    }

    /// True once the effect has run its course and can be dropped.
    pub fn finished(&self) -> bool { self.elapsed >= self.params.total_secs() }

    /// True while the membrane is still growing (pre-snap).
    fn inflating(&self) -> bool { self.elapsed < self.params.inflate_secs }

    /// Membrane phase 0..=1. For `Depart` this runs 0 to 1 as the bubble
    /// swells; for `Arrive` it is mirrored so the effect plays in
    /// reverse — a point swelling out of nothing.
    fn membrane_t(&self) -> f32 {
        let t = (self.elapsed / self.params.inflate_secs.max(1e-4)).clamp(0.0, 1.0);
        match self.dir {
            Direction::Depart => t,
            // Arrival: the membrane collapses inward as the ship
            // materialises, so run the curve backwards.
            Direction::Arrive => 1.0 - t,
        }
    }

    /// Whether the travelling object should be drawn this frame.
    ///
    /// Departure hides it from the snap onward — the pop *causes* the
    /// disappearance rather than trailing it. Arrival hides it until
    /// the snap, so the ship appears exactly as the bubble bursts.
    pub fn ship_hidden(&self) -> bool {
        match self.dir {
            Direction::Depart => !self.inflating(),
            Direction::Arrive => self.inflating(),
        }
    }

    /// Advance the timer and, on the frame the membrane lets go, spawn
    /// the burst. Returns `true` on exactly the snap frame so callers
    /// can fire audio / camera shake in sympathy.
    pub fn update(
        &mut self,
        world: &mut World,
        rand: &mut dyn FnMut() -> f32,
        dt: f32,
    ) -> bool {
        let was_inflating = self.inflating();
        self.elapsed += dt;

        // Feed motes: a thin ring of particles drifting *inward* along
        // the membrane while it inflates. `burst` is radial-outward
        // only, so these are placed by hand on the rim with inward
        // velocity. Spawned a few per frame rather than all at once so
        // the membrane looks continuously fed.
        if self.inflating() && self.params.feed_count > 0 {
            let per_frame = (self.params.feed_count as f32 * dt
                / self.params.inflate_secs.max(1e-4)).max(0.0);
            let n = per_frame.floor() as u32
                + u32::from(rand() < per_frame.fract());
            let r = self.membrane_radius();
            for _ in 0..n {
                let a = rand() * TAU;
                let dir = DVec2::new(a.cos() as f64, a.sin() as f64);
                let spawn = self.pos + dir * (r * 1.15) as f64;
                // Drift inward toward the ship, slowly — these are
                // being drawn *into* the bubble.
                let speed = (r as f64 * 0.9) + rand() as f64 * (r as f64 * 0.6);
                let vel = -dir * speed;
                let life = self.params.inflate_secs * (0.35 + rand() * 0.3);
                let c = self.params.color;
                spawn_particle(
                    world, spawn, vel,
                    [c[0], c[1], c[2], 0.85],
                    [c[0], c[1], c[2], 0.0],
                    life, 2.5, 0.5,
                );
            }
        }

        // The snap: exactly the frame the membrane phase ends.
        if was_inflating && !self.inflating() && !self.snapped {
            self.snapped = true;
            self.spawn_burst(world, rand);
            return true;
        }
        false
    }

    /// Current membrane radius in world units. Eases out so the bubble
    /// decelerates as it approaches its limit — the pause right before
    /// the snap is what sells the tension.
    fn membrane_radius(&self) -> f32 {
        let t = self.membrane_t();
        // Ease-out cubic.
        let eased = 1.0 - (1.0 - t).powi(3);
        self.params.radius * eased
    }

    /// Membrane surface alpha. Thins as the bubble grows — the surface
    /// is being stretched over more area, so there is less of it per
    /// unit length. This is the core of the "about to pop" read.
    fn membrane_alpha(&self) -> f32 {
        let t = self.membrane_t();
        // Fade in fast from nothing, then thin steadily toward the pop.
        let fade_in = (t * 6.0).clamp(0.0, 1.0);
        fade_in * (1.0 - t).powf(0.65) * 0.85
    }

    /// The outward rim burst — fired once, on the snap frame. Particles
    /// originate on the membrane's circumference (not the centre) so
    /// the burst reads as the *surface* letting go.
    fn spawn_burst(&self, world: &mut World, rand: &mut dyn FnMut() -> f32) {
        let r = self.params.radius;
        let (smin, sjit) = self.params.burst_speed;
        let c = self.params.color;
        let h = self.params.hot_color;
        let n = self.params.burst_count;
        if n == 0 { return; }

        for i in 0..n {
            // Even angular spread with jitter so the rim does not look
            // like a stamped cog.
            let a = i as f64 * std::f64::consts::TAU / n as f64
                + rand() as f64 * 0.5;
            let dir = DVec2::new(a.cos(), a.sin());
            // Start on the membrane, with a little radial scatter.
            let start = self.pos + dir * (r as f64 * (0.92 + rand() as f64 * 0.16));
            let speed = smin + rand() as f64 * sjit;
            let vel = dir * speed;
            let life = 0.25 + rand() * 0.45;
            // Alternate hot/cool so the burst has some internal
            // temperature variation rather than one flat hue.
            let hot = i % 3 == 0;
            let (cs, size) = if hot {
                ([h[0], h[1], h[2], 1.0], 5.0 + rand() * 3.0)
            } else {
                ([c[0], c[1], c[2], 0.95], 3.0 + rand() * 2.0)
            };
            spawn_particle(
                world, start, vel, cs,
                [c[0] * 0.4, c[1] * 0.4, c[2] * 0.5, 0.0],
                life, size, size * 0.25,
            );
        }

        // A tight white-hot core flash at the centre — short, fast,
        // shrinking. Borrowed straight from the ship-death "flash"
        // layer, which is the established idiom for a detonation
        // instant in this codebase.
        let core = (n / 3).max(6);
        for i in 0..core {
            let a = i as f64 * std::f64::consts::TAU / core as f64 + rand() as f64;
            let dir = DVec2::new(a.cos(), a.sin());
            let speed = r as f64 * (0.6 + rand() as f64 * 1.2);
            spawn_particle(
                world, self.pos, dir * speed,
                [1.0, 1.0, 1.0, 1.0],
                [h[0], h[1], h[2], 0.0],
                0.16 + rand() * 0.12,
                r * 0.16, r * 0.03,
            );
        }
    }

    /// Draw the membrane + afterglow. `to_screen` converts a world
    /// position to screen space; `scale` converts world units to
    /// pixels (the caller's render scale).
    ///
    /// The renderer offers flat vertex colour only, so the translucent
    /// membrane is faked with a soft fill plus a few concentric
    /// outlines at graded alpha — cheap, and close enough to a
    /// rim-lit shell at gameplay distances.
    pub fn draw(&self, batch: &mut Batch, to_screen: impl Fn(DVec2) -> Vec2, scale: f32) {
        let center = to_screen(self.pos);
        let c = self.params.color;

        if self.inflating() {
            let r = self.membrane_radius() * scale;
            let a = self.membrane_alpha();
            if r <= 0.5 || a <= 0.002 { return; }

            // Interior wash — barely there. A soap bubble is mostly
            // *hole*: the eye reads the rim, not the fill. Anything
            // heavier turns the ship into a silhouette behind a disc.
            batch.circle(center, r, [c[0], c[1], c[2], a * 0.05], 32);

            // The membrane proper: three concentric outlines at
            // falling alpha give the surface a little thickness and a
            // soft outer edge without any gradient support.
            batch.circle_outline(center, r,         2.0, [c[0], c[1], c[2], a], 48);
            batch.circle_outline(center, r * 0.965, 1.0,
                [c[0], c[1], c[2], a * 0.45], 40);
            batch.circle_outline(center, r * 1.03,  1.0,
                [c[0], c[1], c[2], a * 0.30], 40);

            // Highlight arc — a short bright sweep on the upper-left
            // of the shell, the classic cue that a surface is curved
            // and glassy. Rotates slowly so it never looks stamped on.
            let sweep = TAU * 0.16;
            let base = TAU * 0.375 + self.elapsed * 0.6;
            batch.arc(center, r, base, sweep, 2.0, [1.0, 1.0, 1.0, a * 0.5], 10);
        } else if self.params.afterglow_secs > 0.0 {
            // Afterglow: the hole the bubble left, expanding and
            // fading fast. Thin — the particles are doing the work
            // here, this just keeps a trace of the membrane's shape
            // for a few frames so the eye tracks where it went.
            let t = ((self.elapsed - self.params.inflate_secs)
                / self.params.afterglow_secs).clamp(0.0, 1.0);
            let r = self.params.radius * scale * (1.0 + t * 0.7);
            let a = (1.0 - t).powi(2) * 0.55;
            if a > 0.002 {
                batch.circle_outline(center, r, 2.0, [c[0], c[1], c[2], a], 48);
                batch.circle_outline(center, r * 0.9, 1.0,
                    [1.0, 1.0, 1.0, a * 0.4], 40);
            }
        }
    }

    /// Flash strength for this frame, `None` when there is nothing to
    /// emit. Peaks on the snap and decays across the afterglow.
    ///
    /// Returns `(color, radius_world, intensity)`, where `intensity` is
    /// a 0..~N brightness scalar.
    ///
    /// # Choosing a consumer
    ///
    /// This is deliberately a plain value rather than a
    /// `Renderer::push_light` call, because the light pipeline is only
    /// the right consumer for an already-lit scene.
    /// `render_lights_and_composite` *multiplies* everything drawn
    /// before it by a dark-by-default light map — that is exactly what
    /// an interior wants (dark until a fixture lights it) and exactly
    /// what a starfield does not (it would black out the sky and every
    /// ship, leaving only the bubble visible). A caller rendering an
    /// unlit scene should feed this to [`BubblePop::draw_flash`]
    /// instead, which paints the same falloff as additive geometry.
    pub fn flash(&self) -> Option<([f32; 3], f32, f32)> {
        if self.params.flash_intensity <= 0.0 { return None; }
        let p = &self.params;
        let intensity = if self.inflating() {
            // A gentle glow builds under the membrane as it charges.
            let t = self.membrane_t();
            p.flash_intensity * 0.18 * t * t
        } else {
            // Snap: full brightness, decaying quickly.
            let t = ((self.elapsed - p.inflate_secs)
                / p.afterglow_secs.max(1e-4)).clamp(0.0, 1.0);
            p.flash_intensity * (1.0 - t).powi(3)
        };
        if intensity <= 0.001 { return None; }
        let radius = p.radius * p.flash_radius_mul;
        // Blend membrane toward hot as the snap approaches. Capped well
        // below 1 so the glow keeps its drive-blue / gate-cyan identity
        // instead of desaturating to a neutral white wash — the tint is
        // what distinguishes a gate transit from a drive jump at a
        // glance. The white-hot punch is supplied by the small core
        // disc and the burst particles, not by bleaching the halo.
        let mix = if self.inflating() { self.membrane_t() * 0.25 } else { 0.45 };
        let col = [
            p.color[0] + (p.hot_color[0] - p.color[0]) * mix,
            p.color[1] + (p.hot_color[1] - p.color[1]) * mix,
            p.color[2] + (p.hot_color[2] - p.color[2]) * mix,
        ];
        Some((col, radius, intensity))
    }

    /// Paint [`BubblePop::flash`] as additive geometry, for scenes that
    /// do not run the light pipeline (see that method for why a
    /// starfield must not).
    ///
    /// Radial falloff is approximated by stacking a handful of
    /// translucent discs of decreasing radius: with alpha blending,
    /// overlapping discs accumulate toward the centre, so the sum
    /// reads as a soft glow even though every disc is flat-shaded.
    /// Rings are quadratically spaced so the core stays tight instead
    /// of smearing into a flat plate.
    pub fn draw_flash(&self, batch: &mut Batch, to_screen: impl Fn(DVec2) -> Vec2, scale: f32) {
        let Some((col, radius, intensity)) = self.flash() else { return; };
        let center = to_screen(self.pos);
        let r_px = radius * scale;
        if r_px <= 0.5 { return; }

        // Alpha per layer is kept low and the radii fall off steeply, so
        // the outermost ring is a barely-there haze rather than a plate
        // of fog over the whole scene. An early version used a flat
        // per-layer alpha over linearly-spaced radii, which stacked into
        // an almost-opaque disc that greyed out the starfield and buried
        // the burst particles — the glow has to stay *local* to the pop.
        // Many thin layers rather than a few thick ones: with only a
        // handful, each disc edge is individually visible and the glow
        // reads as a set of hard concentric bands (a tree-ring look)
        // instead of a falloff. Stacking ~16 near-transparent discs
        // puts the quantisation below the eye's threshold.
        const LAYERS: u32 = 16;
        // Total accumulated alpha at the centre, budgeted so even the
        // brightest preset stays translucent.
        let budget = (intensity * 0.30).min(0.55);
        let per_layer = budget / LAYERS as f32;
        for i in 0..LAYERS {
            // f: 1.0 at the core, → 0 at the rim.
            let f = 1.0 - (i as f32 / LAYERS as f32);
            // Quadratic radius spacing keeps the stack dense in the
            // middle and lets the outermost disc fade to a wide haze.
            let r = r_px * f * f;
            if r <= 0.5 { continue; }
            batch.circle(center, r, [col[0], col[1], col[2], per_layer], 20);
        }
        // White-hot core right at the snap, so the centre punches
        // through the tinted glow. Small — this is a spark, not a sun.
        if !self.inflating() {
            let core = r_px * 0.055;
            let a = (intensity * 0.35).min(0.85);
            if core > 0.5 {
                batch.circle(center, core, [1.0, 1.0, 1.0, a], 16);
            }
        }
    }
}

/// Thin alias over `particles::spawn_full` — the bubble always spawns
/// on tag 0 and needs explicit control of `color_end` (the motes fade
/// to transparent while holding their hue, which `spawn_sized`'s
/// darken-and-fade default would not give).
#[allow(clippy::too_many_arguments)]
fn spawn_particle(
    world: &mut World,
    pos: DVec2,
    vel: DVec2,
    color_start: [f32; 4],
    color_end: [f32; 4],
    lifetime: f32,
    size_start: f32,
    size_end: f32,
) {
    super::particles::spawn_full(
        world, pos, vel, color_start, color_end, lifetime, 0, size_start, size_end,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{Particle, Transform2D};

    fn params() -> BubbleParams {
        BubbleParams {
            inflate_secs: 0.4,
            afterglow_secs: 0.2,
            radius: 100.0,
            burst_count: 12,
            feed_count: 10,
            burst_speed: (100.0, 100.0),
            color: [0.4, 0.7, 1.0],
            hot_color: [1.0, 1.0, 1.0],
            flash_intensity: 2.0,
            flash_radius_mul: 3.0,
        }
    }

    /// A counter-based stand-in for a real RNG — deterministic so the
    /// particle counts below are exact.
    fn rng() -> impl FnMut() -> f32 {
        let mut n = 0u32;
        move || { n = n.wrapping_add(1); (n % 97) as f32 / 97.0 }
    }

    #[test]
    fn depart_hides_ship_exactly_on_the_snap() {
        let p = params();
        let mut b = BubblePop::new(DVec2::ZERO, Direction::Depart, p);
        assert!(!b.ship_hidden(), "ship visible while the bubble inflates");
        let mut w = World::new();
        let mut r = rng();
        // Step to just before the snap.
        b.update(&mut w, &mut r, p.inflate_secs - 0.01);
        assert!(!b.ship_hidden(), "still visible one frame before the snap");
        // Cross the snap boundary.
        let snapped = b.update(&mut w, &mut r, 0.02);
        assert!(snapped, "update reports the snap frame");
        assert!(b.ship_hidden(), "ship gone on the snap frame");
    }

    #[test]
    fn arrive_is_the_mirror_of_depart() {
        let p = params();
        let mut b = BubblePop::new(DVec2::ZERO, Direction::Arrive, p);
        // Arrival hides the ship until the burst opens.
        assert!(b.ship_hidden());
        let mut w = World::new();
        let mut r = rng();
        b.update(&mut w, &mut r, p.inflate_secs + 0.01);
        assert!(!b.ship_hidden(), "ship present once the bubble bursts open");
    }

    #[test]
    fn snap_fires_once_and_spawns_rim_plus_core() {
        let p = params();
        let mut b = BubblePop::new(DVec2::ZERO, Direction::Depart, p);
        let mut w = World::new();
        let mut r = rng();
        b.update(&mut w, &mut r, p.inflate_secs + 0.01);
        let after_snap = w.iter::<Particle>().count();
        // burst_count rim particles + burst_count/3 core particles, plus
        // whatever feed motes the inflate phase happened to emit.
        let core = (p.burst_count / 3).max(6);
        assert!(after_snap >= (p.burst_count + core) as usize,
            "rim + core particles spawned, got {after_snap}");
        // A second update must not re-fire the burst.
        let again = b.update(&mut w, &mut r, 0.05);
        assert!(!again, "snap is a one-shot");
    }

    #[test]
    fn burst_particles_start_on_the_rim_not_the_centre() {
        let mut p = params();
        // Silence the feed motes so only burst particles exist.
        p.feed_count = 0;
        let mut b = BubblePop::new(DVec2::ZERO, Direction::Depart, p);
        let mut w = World::new();
        let mut r = rng();
        b.update(&mut w, &mut r, p.inflate_secs + 0.01);
        // The rim particles must sit near `radius`; the core ones sit at
        // the centre. At least burst_count of them are out on the rim.
        let on_rim = w.iter::<Transform2D>()
            .filter(|(_, t)| {
                let d = t.pos.length() as f32;
                d > p.radius * 0.85 && d < p.radius * 1.15
            })
            .count();
        assert!(on_rim >= p.burst_count as usize,
            "expected >= {} rim particles, got {on_rim}", p.burst_count);
    }

    #[test]
    fn membrane_thins_as_it_grows() {
        let p = params();
        let mut b = BubblePop::new(DVec2::ZERO, Direction::Depart, p);
        b.elapsed = p.inflate_secs * 0.35;
        let (early_r, early_a) = (b.membrane_radius(), b.membrane_alpha());
        b.elapsed = p.inflate_secs * 0.92;
        let (late_r, late_a) = (b.membrane_radius(), b.membrane_alpha());
        assert!(late_r > early_r, "bubble grows");
        assert!(late_a < early_a, "surface thins as it stretches");
    }

    #[test]
    fn flash_peaks_at_the_snap() {
        let p = params();
        let mut b = BubblePop::new(DVec2::ZERO, Direction::Depart, p);
        b.elapsed = p.inflate_secs * 0.5;
        let charging = b.flash().map(|f| f.2).unwrap_or(0.0);
        b.elapsed = p.inflate_secs + 0.001;
        let at_snap = b.flash().map(|f| f.2).unwrap_or(0.0);
        assert!(at_snap > charging * 4.0,
            "snap flash ({at_snap}) dwarfs the charge glow ({charging})");
        // And it is gone by the end of the afterglow.
        b.elapsed = p.total_secs();
        assert!(b.flash().is_none(), "flash fully decayed");
    }

    #[test]
    fn finishes_after_total_secs() {
        let p = params();
        let mut b = BubblePop::new(DVec2::ZERO, Direction::Depart, p);
        assert!(!b.finished());
        let mut w = World::new();
        let mut r = rng();
        b.update(&mut w, &mut r, p.total_secs() + 0.01);
        assert!(b.finished());
        assert!((b.progress() - 1.0).abs() < 1e-6);
    }
}
