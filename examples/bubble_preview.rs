//! Headless preview for the warp-bubble pop.
//!
//! Rasterises `fx::bubble` frames to PPM images so the effect can be
//! eyeballed without standing up the whole client + server stack. The
//! `Batch` is pure CPU geometry (alpha-blended triangles), so a tiny
//! software rasteriser reproduces exactly what the GPU would draw.
//!
//! Run with:
//!     cargo run -p void_engine --example bubble_preview -- <out_dir>

use glam::{DVec2, Vec2};
use void_engine::World;
use void_engine::fx::bubble::{BubbleParams, BubblePop, Direction};
use void_engine::fx::particles;
use void_engine::components::{Particle, Transform2D};
use void_engine::renderer::batch::Batch;

const W: usize = 480;
const H: usize = 480;

/// Alpha-blend every triangle in the batch into an RGB float buffer.
/// Mirrors the renderer's straight-alpha blend over a dark space
/// background.
fn rasterise(batch: &Batch, buf: &mut [[f32; 3]]) {
    for tri in batch.indices.chunks(3) {
        if tri.len() < 3 { continue; }
        let v: Vec<_> = tri.iter().map(|&i| batch.vertices[i as usize]).collect();
        let (p0, p1, p2) = (
            Vec2::from(v[0].pos), Vec2::from(v[1].pos), Vec2::from(v[2].pos),
        );
        // Screen space is centre-origin, y-up; the buffer is top-left,
        // y-down.
        let to_px = |p: Vec2| Vec2::new(p.x + W as f32 * 0.5, H as f32 * 0.5 - p.y);
        let (a, b, c) = (to_px(p0), to_px(p1), to_px(p2));

        let min_x = a.x.min(b.x).min(c.x).floor().max(0.0) as usize;
        let max_x = (a.x.max(b.x).max(c.x).ceil() as usize).min(W - 1);
        let min_y = a.y.min(b.y).min(c.y).floor().max(0.0) as usize;
        let max_y = (a.y.max(b.y).max(c.y).ceil() as usize).min(H - 1);
        if min_x > max_x || min_y > max_y { continue; }

        let area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
        if area.abs() < 1e-6 { continue; }

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let p = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                // Barycentric inside-test, sign-agnostic so winding
                // order does not matter.
                let w0 = ((b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)) / area;
                let w1 = ((c.x - b.x) * (p.y - b.y) - (c.y - b.y) * (p.x - b.x)) / area;
                let w2 = ((a.x - c.x) * (p.y - c.y) - (a.y - c.y) * (p.x - c.x)) / area;
                if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 { continue; }
                // Flat colour — every vertex of a batch primitive
                // shares one colour, so no interpolation is needed.
                let col = v[0].color;
                let a_ = col[3].clamp(0.0, 1.0);
                let px = &mut buf[y * W + x];
                for k in 0..3 {
                    px[k] = px[k] * (1.0 - a_) + col[k] * a_;
                }
            }
        }
    }
}

/// Rough luminance sum — a cheap proxy for "how bright is this frame",
/// used to print the effect's intensity curve as a sanity check.
fn luma(buf: &[[f32; 3]]) -> f32 {
    buf.iter().map(|p| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]).sum::<f32>()
        / (W * H) as f32
}

fn write_ppm(path: &std::path::Path, buf: &[[f32; 3]]) {
    let mut out = format!("P6\n{W} {H}\n255\n").into_bytes();
    for p in buf {
        for c in p {
            out.push((c.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8);
        }
    }
    std::fs::write(path, out).unwrap();
}

fn render_sequence(name: &str, params: BubbleParams, dir: Direction, out: &std::path::Path) {
    let mut world = World::new();
    let mut seed = 12345u32;
    let mut rand = move || {
        // xorshift — deterministic so previews are reproducible.
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        (seed % 100_000) as f32 / 100_000.0
    };
    let mut pop = BubblePop::new(DVec2::ZERO, dir, params);

    let dt = 1.0 / 60.0;
    let frames = (params.total_secs() / dt).ceil() as u32;
    // Sample ~8 frames across the effect so the contact sheet stays small.
    let stride = (frames / 8).max(1);

    println!("\n=== {name} ({dir:?}) — {frames} frames @60fps ===");
    let mut shot = 0;
    for f in 0..frames {
        let snapped = pop.update(&mut world, &mut rand, dt);
        particles::integrate(&mut world, dt);
        particles::update(&mut world, dt);

        if f % stride != 0 && !snapped { continue; }

        let mut buf = vec![[0.004_f32, 0.006, 0.016]; W * H];
        let mut batch = Batch::new();
        // World units -> pixels. Fit the bubble comfortably in frame.
        let scale = (H as f32 * 0.34) / params.radius;
        let to_screen = |p: DVec2| Vec2::new(p.x as f32, p.y as f32) * scale;

        pop.draw_flash(&mut batch, to_screen, scale);
        pop.draw(&mut batch, to_screen, scale);

        // Particles, drawn the way the client draws them: colour and
        // size both lerped across life.
        for (id, p) in world.iter::<Particle>() {
            let Some(t) = world.get::<Transform2D>(id) else { continue };
            let s = to_screen(t.pos);
            let k = 1.0 - (p.lifetime / p.max_lifetime).clamp(0.0, 1.0);
            let col = [
                p.color_start[0] + (p.color_end[0] - p.color_start[0]) * k,
                p.color_start[1] + (p.color_end[1] - p.color_start[1]) * k,
                p.color_start[2] + (p.color_end[2] - p.color_start[2]) * k,
                p.color_start[3] + (p.color_end[3] - p.color_start[3]) * k,
            ];
            let sz = p.size_start + (p.size_end - p.size_start) * k;
            batch.rect(s, Vec2::splat(sz), col);
        }

        // Stand-in for the ship hull, so the "vanishes on the snap
        // frame" behaviour is visible in the contact sheet.
        if !pop.ship_hidden() {
            batch.rect(Vec2::ZERO, Vec2::splat(24.0 * scale), [0.75, 0.78, 0.85, 1.0]);
        }

        rasterise(&batch, &mut buf);
        let path = out.join(format!("{name}_{shot:02}.ppm"));
        write_ppm(&path, &buf);
        println!(
            "  f{f:<3} t={:.3}s  hull={}  parts={:<4} luma={:.4}{}",
            f as f32 * dt,
            if pop.ship_hidden() { "hidden " } else { "visible" },
            world.iter::<Particle>().count(),
            luma(&buf),
            if snapped { "  <-- SNAP" } else { "" },
        );
        shot += 1;
    }
}

fn main() {
    let out = std::env::args().nth(1).unwrap_or_else(|| ".".into());
    let out = std::path::PathBuf::from(out);
    std::fs::create_dir_all(&out).unwrap();

    use void_claim_presets::*;
    for (name, p) in [
        ("local_warp", local_warp()),
        ("hyperspace", hyperspace()),
        ("gate",       gate()),
    ] {
        render_sequence(name, p, Direction::Depart, &out);
    }
    render_sequence("hyperspace_arrive", hyperspace(), Direction::Arrive, &out);
    println!("\nWrote PPM frames to {}", out.display());
}

/// The client's tuning presets, duplicated here rather than imported —
/// `void_engine` must not depend on a game crate, and this example
/// lives in the engine. Keep in sync with
/// `void_claim::jump_bubble::JumpKind::params`.
mod void_claim_presets {
    use super::BubbleParams;
    const DRIVE_BLUE: [f32; 3] = [0.40, 0.72, 1.00];
    const GATE_CYAN:  [f32; 3] = [0.40, 0.90, 1.00];
    const HOT: [f32; 3] = [1.00, 1.00, 0.96];

    pub fn local_warp() -> BubbleParams {
        BubbleParams {
            inflate_secs: 0.34, afterglow_secs: 0.22, radius: 120.0,
            burst_count: 26, feed_count: 14, burst_speed: (260.0, 340.0),
            color: DRIVE_BLUE, hot_color: HOT,
            flash_intensity: 1.1, flash_radius_mul: 2.4,
        }
    }
    pub fn hyperspace() -> BubbleParams {
        BubbleParams {
            inflate_secs: 0.85, afterglow_secs: 0.45, radius: 320.0,
            burst_count: 84, feed_count: 46, burst_speed: (520.0, 900.0),
            color: DRIVE_BLUE, hot_color: HOT,
            flash_intensity: 2.6, flash_radius_mul: 3.2,
        }
    }
    pub fn gate() -> BubbleParams {
        BubbleParams {
            inflate_secs: 0.55, afterglow_secs: 0.34, radius: 220.0,
            burst_count: 54, feed_count: 30, burst_speed: (380.0, 620.0),
            color: GATE_CYAN, hot_color: [0.85, 1.00, 1.00],
            flash_intensity: 1.9, flash_radius_mul: 2.9,
        }
    }
}
