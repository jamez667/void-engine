//! Generic grid A* pathfinding primitives.
//!
//! [`astar_bool_grid`] takes a `&[bool]` blocking mask and finds a
//! 4-connected path between two cells. Manhattan heuristic, uniform cost
//! per step. Returned path excludes the start cell and includes the goal,
//! in `(col, row)` pairs. Empty `Vec` when start == goal. `None` when
//! unreachable.
//!
//! Game-side grid pathfinders (tile floors with per-tile blocking rules,
//! ACL doors, etc.) live in `void_sim::pathfind` and reuse this
//! primitive when the tile source can be flattened to a bool grid.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// A* open set: a min-heap keyed on `(f_score, g_score, (col, row))`.
/// `Reverse` turns `BinaryHeap`'s max-heap into the min-heap A* wants,
/// and `g_score` breaks f-score ties toward the deeper node.
type OpenSet = BinaryHeap<Reverse<(u32, u32, (i32, i32))>>;

/// A caller-supplied tile-blocking source. Lets `astar_grid` run over
/// any game-side tile representation (station floor, ship exterior,
/// etc.) without engine dragging in game types. Out-of-bounds cells
/// should return `true` from `blocks`.
pub trait TileSource {
    /// Grid dimensions in `(width, height)` tile counts.
    fn dims(&self) -> (u32, u32);
    /// True if `(c, r)` is impassable. Return `true` for out-of-bounds.
    fn blocks(&self, c: i32, r: i32) -> bool;
}

/// A* on any [`TileSource`]. Same rules as [`astar_bool_grid`] —
/// 4-connected, Manhattan heuristic, uniform cost, excludes start,
/// includes goal. `extra_blocked` layers extra impassable cells on top
/// (used to reject e.g. occupied dock pads that the tile source itself
/// would call walkable).
pub fn astar_tile_grid<T: TileSource>(
    src: &T,
    start: (i32, i32),
    goal:  (i32, i32),
    extra_blocked: &HashSet<(i32, i32)>,
) -> Option<Vec<(u16, u16)>> {
    if start == goal { return Some(Vec::new()); }
    if src.blocks(goal.0, goal.1) { return None; }
    let (w, h) = src.dims();
    let heur = |p: (i32, i32)| -> u32 {
        (p.0 - goal.0).unsigned_abs() + (p.1 - goal.1).unsigned_abs()
    };
    let mut open: OpenSet = BinaryHeap::new();
    let mut g_score: HashMap<(i32, i32), u32> = HashMap::new();
    let mut came_from: HashMap<(i32, i32), (i32, i32)> = HashMap::new();
    let mut closed: HashSet<(i32, i32)> = HashSet::new();
    open.push(Reverse((heur(start), 0, start)));
    g_score.insert(start, 0);
    let cap = (w as usize).saturating_mul(h as usize).max(1);
    let mut expanded = 0usize;
    while let Some(Reverse((_f, g, cur))) = open.pop() {
        if cur == goal { return Some(reconstruct(&came_from, cur)); }
        if !closed.insert(cur) { continue; }
        expanded += 1;
        if expanded > cap { return None; }
        for (dx, dy) in [(0i32, 1i32), (0, -1), (1, 0), (-1, 0)] {
            let nb = (cur.0 + dx, cur.1 + dy);
            if src.blocks(nb.0, nb.1) { continue; }
            if extra_blocked.contains(&nb) { continue; }
            if closed.contains(&nb) { continue; }
            let tentative = g + 1;
            let better = g_score.get(&nb).is_none_or(|&old| tentative < old);
            if !better { continue; }
            g_score.insert(nb, tentative);
            came_from.insert(nb, cur);
            open.push(Reverse((tentative + heur(nb), tentative, nb)));
        }
    }
    None
}

/// A* over a caller-supplied bool grid (`blocked[row*width + col]`).
/// 4-connected, Manhattan heuristic. Returned path excludes the start
/// cell and includes the goal, in `(col, row)` pairs.
pub fn astar_bool_grid(
    width:   u32,
    height:  u32,
    blocked: &[bool],
    start:   (i32, i32),
    goal:    (i32, i32),
) -> Option<Vec<(u16, u16)>> {
    if start == goal { return Some(Vec::new()); }
    let w = width as i32;
    let h = height as i32;
    let idx = |c: i32, r: i32| -> Option<usize> {
        if c < 0 || r < 0 || c >= w || r >= h { return None; }
        Some((r as usize) * (width as usize) + (c as usize))
    };
    let blocks = |c: i32, r: i32| -> bool {
        match idx(c, r) {
            Some(i) => blocked.get(i).copied().unwrap_or(true),
            None    => true,
        }
    };
    if blocks(goal.0, goal.1) { return None; }
    let heur = |p: (i32, i32)| -> u32 {
        (p.0 - goal.0).unsigned_abs() + (p.1 - goal.1).unsigned_abs()
    };
    let mut open: OpenSet = BinaryHeap::new();
    let mut g_score: HashMap<(i32, i32), u32> = HashMap::new();
    let mut came_from: HashMap<(i32, i32), (i32, i32)> = HashMap::new();
    let mut closed: HashSet<(i32, i32)> = HashSet::new();
    open.push(Reverse((heur(start), 0, start)));
    g_score.insert(start, 0);
    let cap = (w as usize).saturating_mul(h as usize).max(1);
    let mut expanded = 0usize;
    while let Some(Reverse((_f, g, cur))) = open.pop() {
        if cur == goal { return Some(reconstruct(&came_from, cur)); }
        if !closed.insert(cur) { continue; }
        expanded += 1;
        if expanded > cap { return None; }
        for (dx, dy) in [(0i32, 1i32), (0, -1), (1, 0), (-1, 0)] {
            let nb = (cur.0 + dx, cur.1 + dy);
            if blocks(nb.0, nb.1) { continue; }
            if closed.contains(&nb) { continue; }
            let tentative = g + 1;
            let better = g_score.get(&nb).is_none_or(|&old| tentative < old);
            if !better { continue; }
            g_score.insert(nb, tentative);
            came_from.insert(nb, cur);
            open.push(Reverse((tentative + heur(nb), tentative, nb)));
        }
    }
    None
}

fn reconstruct(
    came_from: &HashMap<(i32, i32), (i32, i32)>,
    mut cur: (i32, i32),
) -> Vec<(u16, u16)> {
    let mut path = Vec::new();
    while let Some(&prev) = came_from.get(&cur) {
        path.push((cur.0 as u16, cur.1 as u16));
        cur = prev;
    }
    path.reverse();
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An open grid: no cell blocked.
    fn open(w: u32, h: u32) -> Vec<bool> {
        vec![false; (w * h) as usize]
    }

    /// A [`TileSource`] over a bool grid, so both entry points are covered by
    /// the same expectations.
    struct Grid {
        w: u32,
        h: u32,
        blocked: Vec<bool>,
    }

    impl TileSource for Grid {
        fn dims(&self) -> (u32, u32) {
            (self.w, self.h)
        }
        fn blocks(&self, c: i32, r: i32) -> bool {
            if c < 0 || r < 0 || c >= self.w as i32 || r >= self.h as i32 {
                return true;
            }
            self.blocked[(r as u32 * self.w + c as u32) as usize]
        }
    }

    #[test]
    fn start_equals_goal_is_an_empty_path() {
        let g = open(4, 4);
        assert_eq!(astar_bool_grid(4, 4, &g, (1, 1), (1, 1)), Some(Vec::new()));
    }

    #[test]
    fn path_excludes_the_start_and_includes_the_goal() {
        let g = open(4, 4);
        let path = astar_bool_grid(4, 4, &g, (0, 0), (2, 0)).expect("reachable");
        assert!(!path.contains(&(0, 0)), "start excluded: {path:?}");
        assert_eq!(path, vec![(1, 0), (2, 0)]);
    }

    #[test]
    fn it_is_4_connected_so_a_diagonal_costs_two_steps() {
        // The property `astar_bool_grid_8` exists to change: no diagonal moves,
        // so (0,0) -> (2,2) is four steps, not two.
        let g = open(4, 4);
        let path = astar_bool_grid(4, 4, &g, (0, 0), (2, 2)).expect("reachable");
        assert_eq!(path.len(), 4);
        for w in path.windows(2) {
            let (a, b) = (w[0], w[1]);
            let d = (a.0 as i32 - b.0 as i32).abs() + (a.1 as i32 - b.1 as i32).abs();
            assert_eq!(d, 1, "orthogonal steps only, got {a:?} -> {b:?}");
        }
    }

    #[test]
    fn a_blocked_goal_is_unreachable() {
        let mut g = open(4, 4);
        g[2 * 4 + 2] = true;
        assert_eq!(astar_bool_grid(4, 4, &g, (0, 0), (2, 2)), None);
    }

    #[test]
    fn an_out_of_bounds_goal_is_unreachable() {
        let g = open(4, 4);
        assert_eq!(astar_bool_grid(4, 4, &g, (0, 0), (9, 9)), None);
    }

    #[test]
    fn it_routes_around_a_wall_through_the_gap() {
        // Column 2 walled except the last row, so the only way across is row 4.
        let (w, h) = (5u32, 5u32);
        let mut g = open(w, h);
        for r in 0..4 {
            g[(r * w + 2) as usize] = true;
        }
        let path = astar_bool_grid(w, h, &g, (0, 0), (4, 0)).expect("gap at row 4");
        assert_eq!(*path.last().unwrap(), (4, 0));
        for &(c, r) in &path {
            assert!(!g[(r as u32 * w + c as u32) as usize], "walked into a wall at {:?}", (c, r));
        }
    }

    #[test]
    fn an_enclosed_goal_is_unreachable() {
        let (w, h) = (5u32, 5u32);
        let mut g = open(w, h);
        for (c, r) in [(3, 0), (3, 1), (3, 2), (4, 2)] {
            g[(r * w + c) as usize] = true;
        }
        assert_eq!(astar_bool_grid(w, h, &g, (0, 0), (4, 0)), None);
    }

    #[test]
    fn the_tile_source_entry_point_agrees_with_the_bool_grid_one() {
        let (w, h) = (6u32, 6u32);
        let mut blocked = open(w, h);
        for r in 1..5 {
            blocked[(r * w + 3) as usize] = true;
        }
        let grid = Grid { w, h, blocked: blocked.clone() };
        let a = astar_tile_grid(&grid, (0, 0), (5, 5), &HashSet::new()).expect("reachable");
        let b = astar_bool_grid(w, h, &blocked, (0, 0), (5, 5)).expect("reachable");
        // Both are optimal under the same rules, so their LENGTHS must match even
        // if tie-breaking picks a different equally-short route.
        assert_eq!(a.len(), b.len(), "{a:?} vs {b:?}");
        assert_eq!(*a.last().unwrap(), (5, 5));
    }

    #[test]
    fn extra_blocked_cells_are_respected() {
        let (w, h) = (4u32, 4u32);
        let grid = Grid { w, h, blocked: open(w, h) };
        // Wall off the goal's whole neighbourhood via `extra_blocked` alone.
        let extra: HashSet<(i32, i32)> = [(2, 3), (3, 2)].into_iter().collect();
        assert_eq!(astar_tile_grid(&grid, (0, 0), (3, 3), &extra), None);
    }
}
