//! Procedural parallax starfield. Stars live in a virtual tile larger
//! than the viewport; a `drift` vector shifts them with per-layer
//! parallax and the result wraps back into the tile so the field always
//! fills the screen. Two draw paths: static twinkling stars and long
//! motion trails (`draw_trails`) for warp-style effects.

use glam::Vec2;
use crate::renderer::batch::Batch;

const STAR_COUNT: usize = 700;

#[derive(Clone, Copy)]
struct Star {
    nx:       f32,   // base position in [-0.5, 0.5)
    ny:       f32,
    layer:    u8,    // 0 = far/dim, 1 = mid, 2 = near/bright
    size:     f32,
    color:    [f32; 3],
    twinkle:  f32,   // phase offset
}

pub struct Starfield {
    stars: Vec<Star>,
}

impl Starfield {
    pub fn new(seed: u32) -> Self {
        let mut s = seed.wrapping_mul(2_166_136_261).wrapping_add(1);
        let mut rng = || {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) & 0x00FF_FFFF) as f32 / 0x0100_0000 as f32
        };
        let stars = (0..STAR_COUNT).map(|_| {
            let layer_r = rng();
            let layer: u8 = if layer_r < 0.55 { 0 } else if layer_r < 0.85 { 1 } else { 2 };
            let bri = match layer {
                0 => 0.30 + rng() * 0.30,
                1 => 0.55 + rng() * 0.30,
                _ => 0.80 + rng() * 0.20,
            };
            let temp = rng();
            let color = if temp < 0.55 {
                [bri * 0.85, bri * 0.92, bri]                // blue-white
            } else if temp < 0.80 {
                [bri,        bri,        bri]                // pure white
            } else if temp < 0.93 {
                [bri,        bri * 0.92, bri * 0.78]         // warm yellow
            } else {
                [bri,        bri * 0.65, bri * 0.45]         // amber/red
            };
            let size = match layer {
                0 => 0.6 + rng() * 0.4,
                1 => 0.9 + rng() * 0.7,
                _ => 1.2 + rng() * 1.0,
            };
            Star {
                nx: rng() - 0.5,
                ny: rng() - 0.5,
                layer,
                size,
                color,
                twinkle: rng() * std::f32::consts::TAU,
            }
        }).collect();
        Self { stars }
    }

    /// Draw the starfield filling `viewport` (centered at origin, +Y up).
    /// `drift` is a screen-pixel offset applied to all stars and scaled by
    /// per-layer parallax; pass `time * direction` for autonomous motion or
    /// `-camera_pos_in_pixels` for camera-locked parallax.
    pub fn draw(&self, batch: &mut Batch, viewport: Vec2, drift: Vec2, time: f32) {
        // Tile slightly larger than the viewport so wrap seams stay off-screen.
        let tile = viewport.max_element().max(64.0) * 1.4;
        let half = tile * 0.5;

        for s in &self.stars {
            let par = match s.layer { 0 => 0.18, 1 => 0.45, _ => 0.85 };
            let bx = s.nx * tile + drift.x * par;
            let by = s.ny * tile + drift.y * par;
            let x = wrap(bx, -half, half);
            let y = wrap(by, -half, half);

            if x.abs() > viewport.x * 0.5 + 4.0 || y.abs() > viewport.y * 0.5 + 4.0 {
                continue;
            }

            // Twinkle: slow shimmer modulating brightness.
            let tw = 0.78 + 0.22 *
                (time * (0.55 + s.layer as f32 * 0.35) + s.twinkle).sin();
            let alpha = match s.layer { 0 => 0.55, 1 => 0.80, _ => 1.00 };
            let core = [s.color[0] * tw, s.color[1] * tw, s.color[2] * tw, alpha];

            // Soft halo for the brightest stars.
            if s.layer >= 2 && s.size > 1.5 {
                let halo = [s.color[0] * tw, s.color[1] * tw, s.color[2] * tw, 0.14];
                batch.rect(Vec2::new(x, y), Vec2::splat(s.size * 2.6), halo);
                let cross = [s.color[0] * tw, s.color[1] * tw, s.color[2] * tw, 0.30];
                batch.rect(Vec2::new(x, y), Vec2::new(s.size * 4.0, 0.6), cross);
                batch.rect(Vec2::new(x, y), Vec2::new(0.6, s.size * 4.0), cross);
            }

            batch.rect(Vec2::new(x, y), Vec2::splat(s.size), core);
        }
    }

    /// Motion trails: stars stay in their wrapped grid positions but each one
    /// gets a tail extending in `flow_dir` — the screen-space direction the
    /// starfield is sliding (i.e. opposite to ship motion). Looks like a
    /// long-exposure photograph as the ship rips through the field. Pass the
    /// same `drift` you'd give to `draw` so the heads line up; the regular
    /// `draw` is intentionally NOT called here so callers can decide whether
    /// to draw stars and trails together (slow → stars only, fast → both).
    pub fn draw_trails(
        &self,
        batch:    &mut Batch,
        viewport: Vec2,
        drift:    Vec2,
        flow_dir: Vec2,
        intensity: f32,
    ) {
        let intensity = intensity.clamp(0.0, 1.0);
        if intensity < 0.01 { return; }
        let tile = viewport.max_element().max(64.0) * 1.4;
        let half = tile * 0.5;

        // Trail base length in pixels — long enough at full intensity to span a
        // good portion of the screen so the motion reads at a glance.
        let base_len = 30.0 + 320.0 * intensity;

        let dir = if flow_dir.length_squared() > 1e-6 {
            flow_dir.normalize()
        } else {
            Vec2::new(1.0, 0.0)
        };

        for s in &self.stars {
            let par = match s.layer { 0 => 0.18, 1 => 0.45, _ => 0.85 };
            let bx = s.nx * tile + drift.x * par;
            let by = s.ny * tile + drift.y * par;
            let x = wrap(bx, -half, half);
            let y = wrap(by, -half, half);

            // Cull trails that start way off-screen (can't possibly intersect).
            if x.abs() > viewport.x * 0.7 + base_len
                || y.abs() > viewport.y * 0.7 + base_len {
                continue;
            }

            // Each star's trail length scales with its layer (parallax) so
            // closer stars leave longer streaks — feels like proper depth.
            let len_mult = match s.layer { 0 => 0.35, 1 => 0.65, _ => 1.0 };
            let len = base_len * len_mult * (0.6 + s.size * 0.4);
            let head = Vec2::new(x, y);      // star's current position (leading edge)
            let tail = head + dir * len;     // streak trails behind

            let layer_alpha = match s.layer { 0 => 0.35, 1 => 0.55, _ => 0.80 };
            let col_head = [
                s.color[0],
                s.color[1],
                s.color[2],
                (intensity * layer_alpha).min(1.0),
            ];
            // Tail end is slightly more saturated blue to read as a velocity smear.
            let col_tail = [
                s.color[0] * 0.55 + 0.10,
                s.color[1] * 0.65 + 0.15,
                s.color[2] * 0.85 + 0.15,
                (intensity * layer_alpha * 0.15).min(1.0),
            ];

            // Two-segment fade: bright head third + dim tail two-thirds.
            // Thick (fast) at head, thin (trailing) at tail.
            let mid = head + dir * (len * 0.35);
            let width_head = (0.8 + s.size * 0.5) * (0.6 + intensity * 0.6) * 0.55;
            let width_tail = width_head * 0.55;
            batch.line(head, mid,  width_head, col_head);
            batch.line(mid,  tail, width_tail, col_tail);
        }
    }
}

fn wrap(v: f32, lo: f32, hi: f32) -> f32 {
    let r = hi - lo;
    if r <= 0.0 { return lo; }
    lo + (v - lo).rem_euclid(r)
}
