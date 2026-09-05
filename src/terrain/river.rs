//! River channels as polylines with a downstream width profile, plus a
//! heightfield-agnostic stream tracer.
//!
//! A river here is geometry, not a simulation: a path of points with a
//! half-width at each vertex, and the spatial queries a renderer or a gameplay
//! system needs against it ("am I in the channel?", "how far to the bank?").
//! Where the path came from is the caller's business — [`trace`] is offered for
//! the common case of walking downhill across a heightfield, but a
//! hand-authored path works just as well.

use crate::rng::Pcg32;
use crate::terrain::field::point_seg_dist2;
use glam::Vec2;

/// One river or tributary: a downhill polyline with a per-vertex channel
/// half-width. Width grows downstream as flow accumulates — a headwater stream
/// widening to a main stem near its mouth.
#[derive(Clone, Debug)]
pub struct River {
    path: Vec<Vec2>,
    half_w: Vec<f32>,
    /// Axis-aligned bounds of the path, for fast spatial rejection.
    bmin: Vec2,
    bmax: Vec2,
}

impl River {
    /// Build a river from a path and a matching half-width per vertex.
    ///
    /// # Panics
    /// If `path` and `half_w` differ in length.
    pub fn new(path: Vec<Vec2>, half_w: Vec<f32>) -> River {
        assert_eq!(path.len(), half_w.len(), "path and half_w must be the same length");
        let mut bmin = Vec2::splat(f32::INFINITY);
        let mut bmax = Vec2::splat(f32::NEG_INFINITY);
        for &p in &path {
            bmin = bmin.min(p);
            bmax = bmax.max(p);
        }
        River { path, half_w, bmin, bmax }
    }

    /// Build a river from a path, widening linearly from `head` at the source to
    /// `mouth` at the last vertex. The common case for a traced stream.
    pub fn tapered(path: Vec<Vec2>, head: f32, mouth: f32) -> River {
        let n = path.len();
        let half_w = (0..n)
            .map(|i| {
                let t = if n > 1 { i as f32 / (n - 1) as f32 } else { 1.0 };
                head + (mouth - head) * t
            })
            .collect();
        River::new(path, half_w)
    }

    pub fn path(&self) -> &[Vec2] {
        &self.path
    }

    pub fn half_widths(&self) -> &[f32] {
        &self.half_w
    }

    /// The river's outflow point, if it has one.
    pub fn mouth(&self) -> Option<Vec2> {
        self.path.last().copied()
    }

    /// Largest half-width along the river — normally its mouth.
    pub fn max_half_w(&self) -> f32 {
        self.half_w.iter().copied().fold(0.0, f32::max)
    }

    /// Squared distance from `pt` to this river's bounding box (0 if inside).
    /// The cheap rejection test that keeps per-segment scans off the hot path.
    #[inline]
    pub fn bbox_dist2(&self, pt: Vec2) -> f32 {
        let dx = (self.bmin.x - pt.x).max(pt.x - self.bmax.x).max(0.0);
        let dy = (self.bmin.y - pt.y).max(pt.y - self.bmax.y).max(0.0);
        dx * dx + dy * dy
    }

    /// Is `pt` inside this channel, accounting for the local width?
    pub fn contains(&self, pt: Vec2) -> bool {
        let mw = self.max_half_w();
        if self.bbox_dist2(pt) > mw * mw {
            return false;
        }
        self.signed_edge_dist(pt) > 0.0
    }

    /// `local_half_width - distance_to_centerline`: positive inside the channel,
    /// negative outside, zero at the bank. A smooth field whose zero-crossing is
    /// the river's edge, so it can be fed straight to a marching-squares tracer
    /// to get bank geometry the same way a coastline is extracted.
    pub fn signed_edge_dist(&self, pt: Vec2) -> f32 {
        let mut best = f32::NEG_INFINITY;
        for i in 0..self.path.len().saturating_sub(1) {
            let (a, b) = (self.path[i], self.path[i + 1]);
            let hw = self.half_w[i].max(self.half_w[i + 1]);
            let signed = hw - point_seg_dist2(pt, a, b).sqrt();
            if signed > best {
                best = signed;
            }
        }
        best
    }

    /// Distance from `pt` to the channel centerline, ignoring width.
    pub fn dist_to_center(&self, pt: Vec2) -> f32 {
        let mut best = f32::INFINITY;
        for i in 0..self.path.len().saturating_sub(1) {
            best = best.min(point_seg_dist2(pt, self.path[i], self.path[i + 1]));
        }
        if best.is_finite() {
            best.sqrt()
        } else {
            f32::INFINITY
        }
    }

    /// A smoothed, densified version of the path for rendering: a Catmull-Rom
    /// spline through the coarse trace vertices, sampled `sub` times per segment,
    /// with half-width interpolated along it. Removes the faceting you would
    /// otherwise see drawing straight chords between widely spaced vertices.
    pub fn smoothed(&self, sub: usize) -> Vec<(Vec2, f32)> {
        let n = self.path.len();
        if n < 2 {
            return self.path.iter().zip(&self.half_w).map(|(&p, &w)| (p, w)).collect();
        }
        let sub = sub.max(1);
        let pt = |i: i32| self.path[i.clamp(0, n as i32 - 1) as usize];
        let hw = |i: i32| self.half_w[i.clamp(0, n as i32 - 1) as usize];
        let mut out = Vec::with_capacity((n - 1) * sub + 1);
        for i in 0..n - 1 {
            let (p0, p1, p2, p3) =
                (pt(i as i32 - 1), pt(i as i32), pt(i as i32 + 1), pt(i as i32 + 2));
            let (w1, w2) = (hw(i as i32), hw(i as i32 + 1));
            // Sample [p1, p2); the final point is appended once after the loop.
            for s in 0..sub {
                let t = s as f32 / sub as f32;
                out.push((catmull_rom(p0, p1, p2, p3, t), w1 + (w2 - w1) * t));
            }
        }
        out.push((self.path[n - 1], self.half_w[n - 1]));
        out
    }
}

/// Catmull-Rom interpolation of `p1`→`p2` (`t` in [0,1]) using neighbours `p0`,
/// `p3` for the tangents. Passes exactly through `p1` and `p2`.
#[inline]
pub fn catmull_rom(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, t: f32) -> Vec2 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

/// A collection of rivers queried as one.
#[derive(Clone, Debug, Default)]
pub struct RiverNetwork {
    rivers: Vec<River>,
}

impl RiverNetwork {
    pub fn new(rivers: Vec<River>) -> Self {
        Self { rivers }
    }

    pub fn rivers(&self) -> &[River] {
        &self.rivers
    }

    pub fn len(&self) -> usize {
        self.rivers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rivers.is_empty()
    }

    pub fn push(&mut self, r: River) {
        self.rivers.push(r);
    }

    /// Is `pt` within any channel?
    pub fn contains(&self, pt: Vec2) -> bool {
        self.rivers.iter().any(|r| r.contains(pt))
    }

    /// Signed edge distance maxed over every channel: positive inside a river,
    /// negative outside, zero at a bank.
    pub fn signed_edge_dist(&self, pt: Vec2) -> f32 {
        let mut best = f32::NEG_INFINITY;
        for r in &self.rivers {
            let mw = r.max_half_w();
            // Only consider channels that could plausibly reach this point.
            if r.bbox_dist2(pt) > (mw * 2.0) * (mw * 2.0) {
                continue;
            }
            best = best.max(r.signed_edge_dist(pt));
        }
        best
    }

    /// Distance to the nearest channel centerline, or infinity if there are none.
    pub fn dist_to_nearest(&self, pt: Vec2) -> f32 {
        self.rivers.iter().map(|r| r.dist_to_center(pt)).fold(f32::INFINITY, f32::min)
    }

    /// The widest river in the network — normally the main stem.
    pub fn largest(&self) -> Option<&River> {
        self.rivers.iter().max_by(|a, b| {
            a.max_half_w().partial_cmp(&b.max_half_w()).unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

/// How [`trace`] should walk a stream downhill.
#[derive(Clone, Debug)]
pub struct TraceParams {
    /// Distance advanced per step, in world units.
    pub step: f32,
    /// Maximum steps before the walk gives up.
    pub max_steps: usize,
    /// How close to an existing channel counts as a confluence.
    pub snap: f32,
    /// How strongly a meander perturbs the path sideways (0 = perfectly straight).
    pub meander: f32,
    /// Number of directions probed when looking for the steepest descent.
    pub probe_dirs: u32,
}

impl Default for TraceParams {
    fn default() -> Self {
        Self { step: 1.0, max_steps: 2000, snap: 3.0, meander: 0.3, probe_dirs: 16 }
    }
}

/// Walk a stream downhill from `start` across the heightfield `height`, stopping
/// at the sea, at a pit, or on reaching an existing channel.
///
/// A pure steepest-descent walk gets trapped: noisy relief is full of local
/// minima, and a stream that falls into one never reaches the coast. `bias`
/// exists to fix that — return a (not necessarily normalized) direction to blend
/// into the descent at a point, typically pointing at the nearest outflow. Its
/// magnitude is its weight, so returning a longer vector nearer the coast makes a
/// river commit to the sea rather than run along it. Return `Vec2::ZERO` for
/// pure steepest descent.
///
/// Returns the path, which is empty if the stream never moved.
pub fn trace(
    start: Vec2,
    p: &TraceParams,
    rng: &mut Pcg32,
    height: impl Fn(Vec2) -> f32,
    sea_level: f32,
    bias: impl Fn(Vec2) -> Vec2,
    existing: &[River],
) -> Vec<Vec2> {
    let jitter = rng.next_u64();
    let mut pos = start;
    let mut path = vec![pos];
    let probe = p.step * 2.0;
    let dirs = p.probe_dirs.max(1);

    for _ in 0..p.max_steps {
        // Merge into an existing channel on contact — that is a confluence.
        if let Some(q) = nearest_vertex_within(existing, pos, p.snap) {
            path.push(q);
            break;
        }

        // Local steepest descent, sampled on a ring around the current point.
        let mut descent = Vec2::ZERO;
        let mut best = height(pos);
        for a in 0..dirs {
            let ang = a as f32 / dirs as f32 * std::f32::consts::TAU;
            let dir = Vec2::new(ang.cos(), ang.sin());
            let e = height(pos + dir * probe);
            if e < best {
                best = e;
                descent = dir;
            }
        }

        let b = bias(pos);
        let mut dir = descent.normalize_or_zero() + b;
        if dir.length_squared() < 1e-6 {
            // A pit with no way down: only the bias can carry the stream out.
            dir = b;
        }
        let dir = dir.normalize_or_zero();
        if dir == Vec2::ZERO {
            break;
        }

        // Gentle meander so channels do not read as straight lines.
        let step_dir = if p.meander != 0.0 {
            let perp = Vec2::new(-dir.y, dir.x);
            let m = super::noise::fbm(pos * 0.0006, 2, jitter) * p.meander;
            (dir + perp * m).normalize_or_zero()
        } else {
            dir
        };
        if step_dir == Vec2::ZERO {
            break;
        }

        pos += step_dir * p.step;
        path.push(pos);

        // Reached open water — the channel stops at the shoreline.
        if height(pos) < sea_level {
            break;
        }
    }

    if path.len() < 2 {
        Vec::new()
    } else {
        path
    }
}

/// The first vertex of any existing river within `snap` of `pos`.
fn nearest_vertex_within(existing: &[River], pos: Vec2, snap: f32) -> Option<Vec2> {
    let s2 = snap * snap;
    for rv in existing {
        if rv.bbox_dist2(pos) > s2 {
            continue;
        }
        for &q in rv.path() {
            if (pos - q).length_squared() < s2 {
                return Some(q);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn straight() -> River {
        River::new(vec![Vec2::new(0.0, 0.0), Vec2::new(100.0, 0.0)], vec![10.0, 10.0])
    }

    #[test]
    fn contains_respects_width() {
        let r = straight();
        assert!(r.contains(Vec2::new(50.0, 5.0)));
        assert!(!r.contains(Vec2::new(50.0, 15.0)));
    }

    #[test]
    fn signed_edge_dist_changes_sign_at_the_bank() {
        let r = straight();
        assert!(r.signed_edge_dist(Vec2::new(50.0, 0.0)) > 0.0, "center must be inside");
        assert!(r.signed_edge_dist(Vec2::new(50.0, 40.0)) < 0.0, "far out must be outside");
        // Exactly on the bank the field is ~0.
        assert!(r.signed_edge_dist(Vec2::new(50.0, 10.0)).abs() < 1e-3);
    }

    #[test]
    fn bbox_rejection_is_conservative() {
        let r = straight();
        assert_eq!(r.bbox_dist2(Vec2::new(50.0, 0.0)), 0.0, "inside the box is zero");
        assert!((r.bbox_dist2(Vec2::new(50.0, 3.0)) - 9.0).abs() < 1e-3);
    }

    #[test]
    fn tapered_widens_downstream() {
        let r =
            River::tapered(vec![Vec2::ZERO, Vec2::new(1.0, 0.0), Vec2::new(2.0, 0.0)], 10.0, 50.0);
        assert_eq!(r.half_widths(), &[10.0, 30.0, 50.0]);
        assert_eq!(r.max_half_w(), 50.0);
        assert_eq!(r.mouth(), Some(Vec2::new(2.0, 0.0)));
    }

    #[test]
    fn smoothed_passes_through_its_control_points() {
        let r =
            River::tapered(vec![Vec2::ZERO, Vec2::new(10.0, 5.0), Vec2::new(20.0, 0.0)], 1.0, 3.0);
        let s = r.smoothed(4);
        assert!(s.len() > 3);
        assert!((s.first().unwrap().0 - Vec2::ZERO).length() < 1e-4);
        assert!((s.last().unwrap().0 - Vec2::new(20.0, 0.0)).length() < 1e-4);
    }

    #[test]
    fn catmull_rom_hits_its_endpoints() {
        let (p0, p1, p2, p3) =
            (Vec2::ZERO, Vec2::new(1.0, 1.0), Vec2::new(2.0, 0.0), Vec2::new(3.0, 1.0));
        assert!((catmull_rom(p0, p1, p2, p3, 0.0) - p1).length() < 1e-5);
        assert!((catmull_rom(p0, p1, p2, p3, 1.0) - p2).length() < 1e-5);
    }

    #[test]
    fn trace_descends_a_slope_and_stops_at_the_sea() {
        // A plane sloping down toward +x, with sea level crossed at x = 100.
        let height = |p: Vec2| 100.0 - p.x;
        let mut rng = Pcg32::seed(1, 1);
        let p = TraceParams { step: 1.0, snap: 0.5, meander: 0.0, ..Default::default() };
        let path = trace(Vec2::ZERO, &p, &mut rng, height, 0.0, |_| Vec2::ZERO, &[]);
        assert!(path.len() > 2, "should have walked downhill");
        assert!(path.last().unwrap().x > 99.0, "ended at {:?}", path.last());
    }

    #[test]
    fn trace_escapes_a_pit_when_biased() {
        // Perfectly flat: pure descent has nowhere to go, so only the bias moves it.
        let height = |_p: Vec2| 5.0;
        let mut rng = Pcg32::seed(2, 1);
        let p = TraceParams {
            step: 1.0,
            snap: 0.5,
            meander: 0.0,
            max_steps: 50,
            ..Default::default()
        };
        let unbiased = trace(Vec2::ZERO, &p, &mut rng, height, 0.0, |_| Vec2::ZERO, &[]);
        assert!(unbiased.is_empty(), "no descent and no bias should not move");

        let biased = trace(Vec2::ZERO, &p, &mut rng, height, 0.0, |_| Vec2::X, &[]);
        assert!(biased.len() > 10, "bias should drive the walk");
        assert!(biased.last().unwrap().x > 10.0);
    }

    #[test]
    fn trace_merges_into_an_existing_channel() {
        let height = |p: Vec2| 100.0 - p.x;
        let mut rng = Pcg32::seed(3, 1);
        // An existing channel crossing the path at x = 20.
        let existing = vec![River::new(
            vec![Vec2::new(20.0, -50.0), Vec2::new(20.0, 0.0), Vec2::new(20.0, 50.0)],
            vec![5.0, 5.0, 5.0],
        )];
        let p = TraceParams { step: 1.0, snap: 2.0, meander: 0.0, ..Default::default() };
        let path = trace(Vec2::ZERO, &p, &mut rng, height, -1000.0, |_| Vec2::ZERO, &existing);
        let end = *path.last().unwrap();
        assert!((end.x - 20.0).abs() < 3.0, "should have merged at the confluence, ended {end:?}");
    }

    #[test]
    fn trace_is_reproducible_for_a_seed() {
        let height = |p: Vec2| 100.0 - p.x;
        let p = TraceParams { step: 1.0, snap: 0.5, ..Default::default() };
        let mut a = Pcg32::seed(9, 9);
        let mut b = Pcg32::seed(9, 9);
        let pa = trace(Vec2::ZERO, &p, &mut a, height, 0.0, |_| Vec2::ZERO, &[]);
        let pb = trace(Vec2::ZERO, &p, &mut b, height, 0.0, |_| Vec2::ZERO, &[]);
        assert_eq!(pa, pb);
    }

    #[test]
    fn network_queries_span_every_river() {
        let net = RiverNetwork::new(vec![
            River::new(vec![Vec2::ZERO, Vec2::new(100.0, 0.0)], vec![5.0, 5.0]),
            River::new(vec![Vec2::new(0.0, 500.0), Vec2::new(100.0, 500.0)], vec![50.0, 50.0]),
        ]);
        assert_eq!(net.len(), 2);
        assert!(net.contains(Vec2::new(50.0, 2.0)));
        assert!(net.contains(Vec2::new(50.0, 520.0)));
        assert!(!net.contains(Vec2::new(50.0, 250.0)));
        assert_eq!(net.largest().unwrap().max_half_w(), 50.0);
        assert!((net.dist_to_nearest(Vec2::new(50.0, 10.0)) - 10.0).abs() < 1e-3);
    }

    #[test]
    fn empty_network_is_inert() {
        let net = RiverNetwork::default();
        assert!(net.is_empty());
        assert!(!net.contains(Vec2::ZERO));
        assert!(net.largest().is_none());
        assert_eq!(net.dist_to_nearest(Vec2::ZERO), f32::INFINITY);
    }
}
