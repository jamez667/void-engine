//! Pinning a 2D tile grid onto a horizontal slice of 3D space, and
//! planning routes across it.
//!
//! # Why the planner stays two-dimensional
//!
//! A walking agent has two degrees of freedom, not three. Its height is
//! decided by gravity and the floor contact, never by the planner, so
//! searching Z would search a dimension the agent cannot choose. The
//! 3D-ness of a walking agent lives entirely in [`NavPlane`], which is
//! the projection between the tile grid and the world.
//!
//! That is also what keeps [`crate::pathfind`]'s Manhattan heuristic
//! admissible. `astar_tile_grid` costs every step the same and measures
//! distance as `|dcol| + |drow|`, which is exact for a 4-connected planar
//! grid — and would stop being a lower bound the moment a vertical move
//! cost something different. A "3D A*" that priced climbing differently
//! would silently return non-optimal paths.
//!
//! [`crate::pathfind::TileSource`] is therefore reused *unchanged*. It
//! never mentioned a dimension: `dims()` and `blocks(c, r)` describe a
//! grid, not a plane in space. There is deliberately no `TileSource3D`.
//!
//! # What this is not
//!
//! There is no support for multiple floors. A building with stairs is
//! served by one `NavPlane` per level plus caller-supplied link tiles,
//! and the search across levels is the caller's. That is a real gap, not
//! an oversight — a voxel search would cost the admissible heuristic
//! above, and a level graph is the standard answer.

use std::collections::HashSet;

use glam::{DVec2, DVec3};

use crate::pathfind::{self, TileSource};

/// How a 2D tile grid is pinned onto a horizontal slice of 3D space.
///
/// # The grid convention, and the trap in it
///
/// The mapping delegates to [`crate::tile_collide::tile_center_default`]
/// rather than reimplementing it, so a 2D walker and a 3D agent standing
/// on the same floor agree about which tile they occupy — byte for byte,
/// not approximately.
///
/// That convention is "grid centred on origin, **row 0 at the top**,
/// +Y up", which means **row increases as world Y decreases**. An agent
/// walking toward a higher row walks toward `-Y`. Every `(col, row)` to
/// world conversion must go through [`NavPlane::tile_center`]; ad-hoc
/// arithmetic that assumes row and Y run the same way sends the agent
/// north to reach a goal to its south, and does it silently on any grid
/// that happens to be square and symmetric.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NavPlane {
    /// Tile grid dimensions, `(width, height)` in tiles.
    pub dims: (u32, u32),
    /// Edge length of one tile, in metres.
    pub tile_size_m: f32,
    /// World Z of the walkable surface — the floor's *top* face, which is
    /// where an agent's feet sit, not the centre of the floor body.
    pub floor_z: f64,
}

impl NavPlane {
    pub fn new(dims: (u32, u32), tile_size_m: f32, floor_z: f64) -> Self {
        Self { dims, tile_size_m, floor_z }
    }

    /// World-space centre of a tile, on the plane.
    pub fn tile_center(&self, col: i32, row: i32) -> DVec3 {
        let p = crate::tile_collide::tile_center_default(
            col,
            row,
            self.dims.0,
            self.dims.1,
            self.tile_size_m,
        );
        DVec3::new(p.x, p.y, self.floor_z)
    }

    /// Which tile a world position is over.
    ///
    /// Z is ignored entirely: an agent halfway through a jump is still
    /// over the tile it will land on, and an agent whose *centre* is a
    /// metre above the floor is still standing where it is standing.
    pub fn pos_to_tile(&self, pos: DVec3) -> (i32, i32) {
        crate::tile_collide::pos_to_tile_default(
            DVec2::new(pos.x, pos.y),
            self.dims.0,
            self.dims.1,
            self.tile_size_m,
        )
    }

    /// The horizontal part of a vector, with Z dropped.
    ///
    /// The one place the "walking is planar" assumption is written down.
    /// Steering flattens both the direction to the target and the agent's
    /// own velocity through this, so a walk force can never acquire a
    /// vertical component and an agent can never try to fly to a goal
    /// below it.
    pub fn flatten(&self, v: DVec3) -> DVec3 {
        DVec3::new(v.x, v.y, 0.0)
    }

    /// Which tiles a set of obstacles covers, as `extra_blocked` cells for
    /// [`plan_path`].
    ///
    /// This is the planning half of what [`crate::avoid`] does reactively.
    /// A plan routes *around* what is known to be in the way; a steer
    /// reacts to whatever turns up. An agent with only the steer walks
    /// into culs-de-sac a plan would have avoided, and an agent with only
    /// the plan walks into anything that moved since it planned. The two
    /// take the same [`crate::avoid::Obstacle`] list so a caller gathers
    /// its surroundings once and spends them on both.
    ///
    /// `self_radius` is the agent's own half-width. It is added to every
    /// obstacle's radius, which is what makes the route one an agent of
    /// that width can actually fit through rather than one only its
    /// centre-point fits through — the difference between a path and a
    /// path that wedges the agent in a doorway.
    ///
    /// # Why a disc and not a bounding box
    ///
    /// The cheap version blocks the axis-aligned tile rectangle around
    /// each obstacle. That over-blocks the corners by a factor of
    /// `4/pi`, and on a coarse grid a single crate can wall off a
    /// diagonal gap the agent would fit through. Testing the tile centre
    /// against the disc costs one extra multiply per candidate cell and
    /// blocks what is actually covered.
    ///
    /// Tiles whose centres fall outside the grid are dropped: a
    /// `HashSet` of cells that do not exist would be carried through the
    /// search and matched against nothing.
    pub fn blocked_cells_from(
        &self,
        obstacles: &[crate::avoid::Obstacle],
        self_radius: f64,
    ) -> HashSet<(i32, i32)> {
        let mut cells = HashSet::new();
        let tile = self.tile_size_m as f64;
        if tile <= 0.0 {
            return cells;
        }

        for o in obstacles {
            let reach = o.radius + self_radius;
            if reach <= 0.0 {
                continue;
            }

            // The tile the obstacle's centre sits on, then outward far
            // enough to cover `reach`. The `+ 1` is the half-tile the
            // centre can sit off-centre within its own tile: without it
            // an obstacle sitting near a tile edge loses the far row.
            let (cc, cr) = self.pos_to_tile(o.pos);
            let span = (reach / tile).ceil() as i32 + 1;

            for dr in -span..=span {
                for dc in -span..=span {
                    let (c, r) = (cc + dc, cr + dr);
                    if c < 0 || r < 0 || c >= self.dims.0 as i32 || r >= self.dims.1 as i32 {
                        continue;
                    }
                    // Against the tile *centre*, in the plane. Z is
                    // dropped for the same reason `pos_to_tile` ignores
                    // it: a crate stacked two metres up still occupies
                    // the ground the agent would walk through.
                    let centre = self.tile_center(c, r);
                    let dx = centre.x - o.pos.x;
                    let dy = centre.y - o.pos.y;
                    if dx * dx + dy * dy <= reach * reach {
                        cells.insert((c, r));
                    }
                }
            }
        }
        cells
    }
}

/// A plan: the tiles left to walk, and the plane they are on.
#[derive(Clone, Debug, PartialEq)]
pub struct NavPath {
    pub plane: NavPlane,
    /// The route, in `(col, row)`. Excludes the tile the agent started
    /// on and includes the goal tile, inheriting
    /// [`crate::pathfind::astar_tile_grid`]'s contract.
    pub waypoints: Vec<(i32, i32)>,
    /// How far into `waypoints` the agent has walked.
    pub cursor: usize,
}

impl NavPath {
    /// The tile the agent is currently heading for.
    pub fn next_waypoint(&self) -> Option<(i32, i32)> {
        self.waypoints.get(self.cursor).copied()
    }

    /// Where that tile is in the world.
    pub fn next_world(&self) -> Option<DVec3> {
        self.peek_world(0)
    }

    /// Where the waypoint `ahead` places further along the route is.
    ///
    /// `peek_world(0)` is the next waypoint. Looking one further ahead is
    /// what lets a caller tell "this waypoint is in front of me" from
    /// "this waypoint is behind me and I should skip it" — a distance
    /// test alone cannot, because both look the same from far away.
    pub fn peek_world(&self, ahead: usize) -> Option<DVec3> {
        self.waypoints
            .get(self.cursor + ahead)
            .map(|&(c, r)| self.plane.tile_center(c, r))
    }

    /// Whether the next waypoint is the last one.
    ///
    /// Steering brakes on the final leg only; slowing into every tile
    /// corner produces an agent that shuffles rather than walks.
    pub fn on_final_leg(&self) -> bool {
        self.cursor + 1 >= self.waypoints.len()
    }

    pub fn advance(&mut self) {
        self.cursor = (self.cursor + 1).min(self.waypoints.len());
    }

    pub fn is_complete(&self) -> bool {
        self.cursor >= self.waypoints.len()
    }

    pub fn remaining(&self) -> usize {
        self.waypoints.len().saturating_sub(self.cursor)
    }
}

/// Plan a route across `plane`, between two world positions.
///
/// Wraps [`crate::pathfind::astar_tile_grid`]; `src` carries the caller's
/// own blocking rules and `extra_blocked` layers transient obstacles on
/// top without rebuilding it — an occupied crate tile, say.
///
/// Returns `None` for the reasons A* does: the goal is blocked, off-grid,
/// or unreachable. A caller should treat that as "there is no route",
/// not as an error.
///
/// # On agents wider than a tile
///
/// The path is a sequence of tile centres, and nothing here knows how
/// wide the agent is. A 4-connected route can clip the corner of a wall
/// that neither waypoint is inside, and a wide agent then grinds against
/// it. The fix costs nothing and belongs in the caller's `blocks`: return
/// true for any tile *adjacent* to a wall when the agent is wider than
/// half a tile. Path smoothing would also fix it and is not built here.
pub fn plan_path<T: TileSource>(
    plane: NavPlane,
    src: &T,
    from: DVec3,
    to: DVec3,
    extra_blocked: &HashSet<(i32, i32)>,
) -> Option<NavPath> {
    let start = plane.pos_to_tile(from);
    let goal = plane.pos_to_tile(to);
    let waypoints = pathfind::astar_tile_grid(src, start, goal, extra_blocked)?;
    Some(NavPath { plane, waypoints, cursor: 0 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avoid::Obstacle;

    /// A non-square grid on purpose: a square one hides a swapped
    /// col/row, and a symmetric one hides a flipped row axis.
    fn plane() -> NavPlane {
        NavPlane::new((8, 5), 2.0, 0.5)
    }

    /// An open floor, so pathing tests exercise the mapping rather than
    /// the caller's blocking rules.
    struct OpenFloor(u32, u32);
    impl TileSource for OpenFloor {
        fn dims(&self) -> (u32, u32) {
            (self.0, self.1)
        }
        fn blocks(&self, c: i32, r: i32) -> bool {
            c < 0 || r < 0 || c >= self.0 as i32 || r >= self.1 as i32
        }
    }

    /// A floor split by a full-height wall, with one gap.
    struct WalledFloor;
    impl TileSource for WalledFloor {
        fn dims(&self) -> (u32, u32) {
            (8, 5)
        }
        fn blocks(&self, c: i32, r: i32) -> bool {
            if c < 0 || r < 0 || c >= 8 || r >= 5 {
                return true;
            }
            // A wall down column 4, with no way through at all.
            c == 4
        }
    }

    /// The mapping must be a true inverse, or an agent standing still
    /// disagrees with itself about which tile it is on and oscillates
    /// between two waypoints.
    #[test]
    fn a_tile_round_trips_through_the_world_and_back() {
        let p = plane();
        for col in 0..8 {
            for row in 0..5 {
                let world = p.tile_center(col, row);
                assert_eq!(
                    p.pos_to_tile(world),
                    (col, row),
                    "tile ({col},{row}) did not survive the round trip",
                );
            }
        }
    }

    /// The 3D plane and the 2D walker must agree about where a tile is.
    ///
    /// They share a floor in any game that has both, and a mapping that
    /// drifted by half a tile would put them on visibly different
    /// squares while every isolated test still passed.
    #[test]
    fn the_nav_plane_agrees_with_the_2d_tile_helpers() {
        let p = plane();
        for (col, row) in [(0, 0), (7, 4), (3, 2), (5, 1)] {
            let got = p.tile_center(col, row);
            let want = crate::tile_collide::tile_center_default(
                col, row, p.dims.0, p.dims.1, p.tile_size_m,
            );
            assert_eq!((got.x, got.y), (want.x, want.y), "tile ({col},{row})");
            assert_eq!(got.z, p.floor_z, "a waypoint must sit on the floor plane");
        }
    }

    /// Row 0 is at the *top*, so walking to a higher row walks toward
    /// negative Y. This is the convention an agent gets wrong silently:
    /// with the sign flipped it walks north to reach a goal to its south
    /// and every round-trip test above still passes.
    #[test]
    fn increasing_row_walks_toward_negative_y() {
        let p = plane();
        let top = p.tile_center(2, 0);
        let bottom = p.tile_center(2, 4);
        assert!(
            bottom.y < top.y,
            "row 4 should be south of row 0, got y {} then {}",
            top.y,
            bottom.y,
        );
        // And columns run the ordinary way.
        assert!(p.tile_center(7, 2).x > p.tile_center(0, 2).x);
    }

    /// Z is not part of the question. An agent's centre sits a half-height
    /// above the floor, so a mapping that cared about Z would fail to
    /// locate an agent standing on the very tile it is asking about.
    #[test]
    fn the_tile_lookup_ignores_height() {
        let p = plane();
        let on_floor = p.tile_center(3, 1);
        for dz in [0.0, 0.9, 5.0, -3.0] {
            let lifted = on_floor + DVec3::new(0.0, 0.0, dz);
            assert_eq!(p.pos_to_tile(lifted), (3, 1), "height {dz} changed the tile");
        }
    }

    /// Inherits A*'s contract: the start tile is excluded, the goal tile
    /// is included. An agent given its own tile as the first waypoint
    /// would immediately "arrive" at it and skip a step.
    #[test]
    fn a_path_excludes_the_start_tile_and_ends_at_the_goal() {
        let p = plane();
        let path = plan_path(
            p,
            &OpenFloor(8, 5),
            p.tile_center(0, 0),
            p.tile_center(3, 0),
            &HashSet::new(),
        )
        .expect("an open floor is always walkable");

        assert_eq!(path.next_waypoint(), Some((1, 0)), "start tile should be skipped");
        assert_eq!(path.waypoints.last().copied(), Some((3, 0)));
        assert!(!path.is_complete());
    }

    /// Every waypoint sits on the floor plane however high the query
    /// position was, or an airborne agent plans a route through the air.
    #[test]
    fn waypoints_sit_on_the_floor_whatever_height_was_asked_about() {
        let p = plane();
        let high = p.tile_center(0, 0) + DVec3::new(0.0, 0.0, 12.0);
        let path = plan_path(p, &OpenFloor(8, 5), high, p.tile_center(4, 3), &HashSet::new())
            .expect("route exists");
        for (c, r) in &path.waypoints {
            assert_eq!(p.tile_center(*c, *r).z, p.floor_z);
        }
    }

    /// No route means no path, so the caller can say "blocked" rather
    /// than walking an agent into a wall forever.
    #[test]
    fn an_unreachable_goal_produces_no_path() {
        let p = plane();
        assert!(
            plan_path(
                p,
                &WalledFloor,
                p.tile_center(0, 0),
                p.tile_center(7, 0),
                &HashSet::new(),
            )
            .is_none(),
            "a solid wall should make the far side unreachable",
        );
    }

    /// `extra_blocked` is how a caller says "a crate is standing there"
    /// without rebuilding its tile source. The seam the stacking
    /// behaviour will need.
    #[test]
    fn transient_obstacles_divert_the_route() {
        let p = plane();
        let open = OpenFloor(8, 5);
        let direct = plan_path(p, &open, p.tile_center(0, 0), p.tile_center(2, 0), &HashSet::new())
            .expect("route exists");
        assert!(direct.waypoints.contains(&(1, 0)));

        let blocked: HashSet<(i32, i32)> = [(1, 0)].into_iter().collect();
        let around = plan_path(p, &open, p.tile_center(0, 0), p.tile_center(2, 0), &blocked)
            .expect("there is a way around");
        assert!(
            !around.waypoints.contains(&(1, 0)),
            "the route should avoid the blocked tile, got {:?}",
            around.waypoints,
        );
    }

    /// The cursor must reach the end and stop there; an `advance` past
    /// the last waypoint must not run the index off the array.
    #[test]
    fn walking_the_cursor_off_the_end_completes_the_path() {
        let p = plane();
        let mut path = plan_path(
            p,
            &OpenFloor(8, 5),
            p.tile_center(0, 0),
            p.tile_center(2, 0),
            &HashSet::new(),
        )
        .expect("route exists");

        assert!(path.remaining() > 0);
        for _ in 0..10 {
            path.advance();
        }
        assert!(path.is_complete());
        assert_eq!(path.next_waypoint(), None);
        assert_eq!(path.remaining(), 0);
    }

    /// Only the last leg is final, or the agent brakes into every tile
    /// corner and shuffles instead of walking.
    #[test]
    fn only_the_last_waypoint_counts_as_the_final_leg() {
        let p = plane();
        let mut path = plan_path(
            p,
            &OpenFloor(8, 5),
            p.tile_center(0, 0),
            p.tile_center(3, 0),
            &HashSet::new(),
        )
        .expect("route exists");

        assert!(!path.on_final_leg(), "three waypoints remain");
        path.advance();
        assert!(!path.on_final_leg(), "two waypoints remain");
        path.advance();
        assert!(path.on_final_leg(), "the last waypoint is the final leg");
    }

    /// An obstacle blocks the tile it stands on.
    ///
    /// The floor of the whole feature: if the cell an obstacle occupies
    /// is not blocked, nothing downstream routes around anything.
    #[test]
    fn an_obstacle_blocks_the_tile_it_stands_on() {
        let p = plane();
        let at = p.tile_center(3, 2);
        let cells = p.blocked_cells_from(&[Obstacle::new(at, 0.3)], 0.0);
        assert!(
            cells.contains(&(3, 2)),
            "the obstacle's own tile is not blocked, got {cells:?}",
        );
    }

    /// A small obstacle blocks *only* its own tile.
    ///
    /// The other half of the previous test: a rasteriser that blocks a
    /// fixed rectangle passes that one and fails this. On a 2 m grid a
    /// crate is small enough to walk past, and an agent that treats every
    /// crate as a 3x3 block of wall will report no route across a yard
    /// that is mostly empty.
    #[test]
    fn a_small_obstacle_does_not_wall_off_its_neighbours() {
        let p = plane();
        let at = p.tile_center(3, 2);
        let cells = p.blocked_cells_from(&[Obstacle::new(at, 0.3)], 0.0);
        assert_eq!(
            cells.len(),
            1,
            "a 0.3 m obstacle on a 2 m grid spread beyond its own tile: {cells:?}",
        );
    }

    /// The agent's own width is added to the obstacle's.
    ///
    /// A route planned for a point is not a route a body can walk. This
    /// is what stops the planner threading the agent through a gap its
    /// shoulders do not fit.
    #[test]
    fn the_agent_radius_widens_what_is_blocked() {
        let p = plane();
        let at = p.tile_center(3, 2);
        let narrow = p.blocked_cells_from(&[Obstacle::new(at, 0.3)], 0.0);
        let wide = p.blocked_cells_from(&[Obstacle::new(at, 0.3)], 2.0);
        assert!(
            wide.len() > narrow.len(),
            "a 2 m-wide agent blocked no more than a point: {} vs {}",
            wide.len(),
            narrow.len(),
        );
        assert!(
            wide.is_superset(&narrow),
            "widening the agent unblocked a cell that was blocked",
        );
    }

    /// Blocked cells are a disc, not the bounding box around it.
    ///
    /// The corner of the square that circumscribes a disc is
    /// `sqrt(2)` times further out than its edge. Blocking that corner
    /// costs a diagonal gap the agent would fit through, which on a
    /// coarse grid is the difference between a route and "no route".
    #[test]
    fn the_blocked_region_is_a_disc_not_a_square() {
        let p = plane();
        let at = p.tile_center(4, 2);
        // Reach past the edge-adjacent tiles but not past the diagonal
        // ones: 2.0 m clears a neighbour centre, 2.83 m would be needed
        // for a corner.
        let cells = p.blocked_cells_from(&[Obstacle::new(at, 2.0)], 0.0);
        assert!(cells.contains(&(5, 2)), "the tile due east is within reach");
        assert!(cells.contains(&(4, 1)), "the tile due north is within reach");
        assert!(
            !cells.contains(&(5, 1)),
            "the diagonal corner is 2.83 m away and must stay open, got {cells:?}",
        );
    }

    /// Tiles off the edge of the grid are dropped.
    ///
    /// They would be carried through the search and matched against
    /// nothing — and `astar_tile_grid` already refuses to leave the grid,
    /// so they are pure weight.
    #[test]
    fn cells_outside_the_grid_are_not_reported() {
        let p = plane();
        let corner = p.tile_center(0, 0);
        let cells = p.blocked_cells_from(&[Obstacle::new(corner, 4.0)], 0.0);
        assert!(!cells.is_empty(), "an obstacle on the corner blocks something");
        for &(c, r) in &cells {
            assert!(
                c >= 0 && r >= 0 && c < p.dims.0 as i32 && r < p.dims.1 as i32,
                "reported off-grid cell ({c}, {r})",
            );
        }
    }

    /// The whole point: a planned route goes *round* an obstacle.
    ///
    /// The unit tests above pin the rasterising; this pins that it is
    /// wired to the planner at all, and is the one that fails if
    /// `extra_blocked` is ever dropped on the way through.
    #[test]
    fn a_planned_route_detours_around_an_obstacle() {
        let p = NavPlane::new((8, 5), 2.0, 0.5);
        let src = OpenFloor(8, 5);
        let start = p.tile_center(0, 2);
        let goal = p.tile_center(7, 2);

        let clear = plan_path(p, &src, start, goal, &HashSet::new())
            .expect("an open floor has a route");
        assert!(
            clear.waypoints.iter().all(|&(_, r)| r == 2),
            "the unobstructed route should run straight along row 2, got {:?}",
            clear.waypoints,
        );

        // A crate squarely in the middle of that straight line.
        let blocked = p.blocked_cells_from(&[Obstacle::new(p.tile_center(4, 2), 0.5)], 0.5);
        let detour = plan_path(p, &src, start, goal, &blocked)
            .expect("there is room to walk round one crate");

        assert!(
            detour.waypoints.iter().any(|&(_, r)| r != 2),
            "the route ran straight through the crate: {:?}",
            detour.waypoints,
        );
        assert!(
            detour.waypoints.iter().all(|w| !blocked.contains(w)),
            "the route passes through a blocked cell",
        );
    }
}
