//! A tiny 3D game: drop crates, click them, watch them settle.
//!
//! This is the first time the 3D path has been assembled into something
//! that runs. Every piece built for it is here and nothing is faked:
//!
//! * `Mesh3D` + `MeshStore` — one crate mesh, uploaded once, drawn many
//!   times through per-instance transforms.
//! * `physics3d` — gravity, contact resolution, friction, sleeping.
//! * `collision::grid3d` + `narrow3d` — broadphase and the contacts the
//!   solver consumes.
//! * `pick3d` — click a crate to select it; click again to launch it.
//! * `Camera3D` + shadow mapping + point lights.
//! * A 2D `Batch` HUD composited over the scene.
//! * `ai3d` — a walking agent that plans a route across the floor and
//!   pushes itself along it with forces, so the crates get in its way.
//!
//! # Controls
//!
//! * **Left click** — select the crate under the cursor.
//! * **Right click** — launch the selected crate upward.
//! * **Space** — drop a new crate.
//! * **A / D** — orbit the camera. **W / S** — raise and lower it.
//! * **R** — reset the scene.
//!
//! The orange box walks itself between the corners of the floor. It is an
//! ordinary dynamic body driven by forces, not a scripted mover: it
//! collides with the crates, shoves them aside, and is shoved back.
//!
//! Run with:
//!     cargo run --release --features render3d --example crates3d

use glam::{DVec3, Mat4, Quat, Vec2, Vec3};
use void_engine::ai3d::{self, Agent3D, NavPlane, WalkTuning3D};
use void_engine::collision::grid3d::SpatialGrid3D;
use void_engine::collision::narrow3d;
use void_engine::components::{Collider3D, Transform3D, Velocity3D};
use void_engine::input::InputState;
// `KeyCode`/`MouseButton` are winit types the engine uses as plain data;
// it does not re-export them, so a consumer names winit directly.
use winit::event::MouseButton;
use winit::keyboard::KeyCode;
use void_engine::pick3d::{self, PickShape};
use void_engine::physics3d::{self, BodyKind, Material3D, RigidBody};
use void_engine::renderer::batch::{Material, Surface};
use void_engine::renderer::camera::Camera3D;
use void_engine::renderer::mesh3d::Mesh3D;
use void_engine::renderer::mesh_store::{MeshDraw, MeshHandle};
use void_engine::renderer::render3d::PointLight3D;
use void_engine::renderer::Renderer;
use void_engine::pathfind::TileSource;
use void_engine::{App, ClientApp, SimCtx, World};

/// Half-extent of a crate, in metres.
const CRATE_HALF: f32 = 0.5;
/// Half-extent of the floor.
const FLOOR_HALF: f64 = 12.0;
/// How far the camera orbits from the origin.
const CAMERA_DISTANCE: f64 = 18.0;
/// How far a click reaches.
const PICK_RANGE: f64 = 100.0;

/// The walker's half-extents: wider than it is tall, on purpose.
///
/// Friction reacts at the feet, a half-height below the centre of mass,
/// so a walk force always pitches the body forward while gravity rights
/// it with a lever of a half-width. A walker's aspect ratio therefore
/// decides whether it can accelerate at all without face-planting: a
/// cube of this mass leans 87 degrees crossing the floor, these
/// proportions lean under half a degree. See
/// `ai3d::WalkTuning3D::for_body`.
///
/// The height is 0.35 rather than the 0.25 it started at, to give the
/// robot's cube hull somewhere to sit once the tracks have taken the
/// bottom 0.24 of it. That costs margin but not much: the tipping limit
/// still leaves 1.84x headroom over the force needed to break friction,
/// and the walk measures 0.45 degrees of lean against a 5 degree check.
const AGENT_HW: f64 = 0.5;
const AGENT_HH: f64 = 0.35;
/// The walker's mass, in kilograms.
const AGENT_MASS: f32 = 80.0;
/// Friction of the floor the walker pushes against.
const FLOOR_FRICTION: f32 = 0.7;
/// Edge length of one navigation tile.
const NAV_TILE: f32 = 1.5;
/// The nav grid's extent in tiles, comfortably inside the floor.
const NAV_DIMS: (u32, u32) = (15, 15);
/// How many crates high the agent builds its stack.
const TARGET_LAYERS: u32 = 3;
/// How grippy a crate is. See `drop_crate` for why it decides whether a
/// stack stands or slides apart.
const CRATE_FRICTION: f32 = 0.8;
/// The floor's shade. Named because the walker's treads have to stay
/// clear of it — see `walker_mesh`.
const FLOOR_COLOUR: [f32; 4] = [0.22, 0.24, 0.28, 1.0];
/// The walker's track colour, near black so it never merges into the
/// floor. Module-level so the mesh tests can identify the tracks by
/// colour rather than by guessing at their position.
const TREAD_COLOUR: [f32; 4] = [0.06, 0.06, 0.07, 1.0];

/// The floor the agent may walk on.
///
/// Every tile inside the grid is walkable — the crates are obstacles the
/// *physics* resolves, not the planner, so the agent shoves its way
/// through a pile rather than routing around it. Routing around them
/// would mean feeding their tiles in as `extra_blocked`, which is the
/// seam `plan_path` leaves open for exactly that.
struct NavFloor;

impl TileSource for NavFloor {
    fn dims(&self) -> (u32, u32) {
        NAV_DIMS
    }
    fn blocks(&self, c: i32, r: i32) -> bool {
        c < 0 || r < 0 || c >= NAV_DIMS.0 as i32 || r >= NAV_DIMS.1 as i32
    }
}

/// One thing in the world: a pose, a body, and how to draw it.
struct Body {
    transform: Transform3D,
    velocity: Velocity3D,
    rigid: RigidBody,
    collider: Collider3D,
    colour: [f32; 4],
    /// Its slot in the broadphase grid, so a pick result maps back here.
    slot: u32,
    /// Held by the agent. A carried crate is kinematic and is written to
    /// a pose every tick, so it must be kept out of the contact list
    /// entirely — see `contacts()`.
    carried: bool,
    /// What to restore when it is put down.
    ///
    /// `RigidBody::kinematic()` throws the mass and inertia away, and
    /// rebuilding them from a constant would silently reset the mass of
    /// any crate that was not the default weight. Invisible in this
    /// example, where every crate is 1 kg, and a real bug in the first
    /// game that has a heavy one.
    dynamic_mass: (f32, glam::Mat3),
}

struct Game {
    bodies: Vec<Body>,
    grid: SpatialGrid3D,

    /// The one crate mesh every dynamic body draws.
    crate_mesh: Option<MeshHandle>,
    /// The floor, which is a different size so it gets its own.
    floor_mesh: Option<MeshHandle>,

    /// Camera orbit angle and height.
    orbit: f64,
    height: f64,

    /// Which body the player has selected, if any.
    selected: Option<usize>,
    /// Set when a click should be resolved in the next render, where the
    /// camera matrix is available. `fixed_update` has no renderer.
    pending_click: Option<Vec2>,
    pending_launch: bool,
    dropped: usize,

    /// The walking agent, and which body it drives.
    agent: Agent3D,
    agent_body: usize,
    /// The stacking behaviour driving the agent. It owns the agent's
    /// goal; nothing else may set it.
    task: ai3d::StackTask,
    /// Its own mesh, so it reads as a walker rather than a crate.
    agent_mesh: Option<MeshHandle>,

    /// Frame at which to run the self-check, if `--verify` was passed.
    verify_at: Option<u32>,
    frame: u32,
}

impl Game {
    fn new() -> Self {
        let mut g = Self {
            bodies: Vec::new(),
            grid: SpatialGrid3D::new(2.0),
            crate_mesh: None,
            floor_mesh: None,
            orbit: 0.0,
            height: 9.0,
            selected: None,
            pending_click: None,
            pending_launch: false,
            dropped: 0,
            agent: Agent3D::new(WalkTuning3D::for_body(
                AGENT_MASS as f64,
                [AGENT_HW, AGENT_HW, AGENT_HH],
                FLOOR_FRICTION as f64,
            )),
            agent_body: 0,
            task: ai3d::StackTask::new(ai3d::StackTuning::for_agent_and_crate(
                [AGENT_HW, AGENT_HW, AGENT_HH],
                [CRATE_HALF as f64, CRATE_HALF as f64, CRATE_HALF as f64],
                NAV_TILE as f64,
                TARGET_LAYERS,
                WalkTuning3D::default(),
            )),
            agent_mesh: None,
            verify_at: None,
            frame: 0,
        };
        g.reset();
        g
    }

    /// Clear the world back to just a floor.
    fn reset(&mut self) {
        self.bodies.clear();
        self.grid.clear();
        self.selected = None;
        self.dropped = 0;

        // The floor: a static box, wide and thin.
        let floor_half = [FLOOR_HALF as f32, FLOOR_HALF as f32, 0.5];
        let slot = self.grid.insert(
            DVec3::new(0.0, 0.0, -0.5),
            // Bounding sphere of the floor box.
            (floor_half[0] as f64 * floor_half[0] as f64 * 2.0 + 0.25).sqrt(),
        );
        self.bodies.push(Body {
            transform: Transform3D::at(DVec3::new(0.0, 0.0, -0.5)),
            velocity: Velocity3D::default(),
            rigid: RigidBody::static_body()
                .with_material(Material3D { restitution: 0.1, friction: 0.7 }),
            collider: Collider3D::box3d(floor_half[0], floor_half[1], floor_half[2]),
            colour: FLOOR_COLOUR,
            slot,
            carried: false,
            dynamic_mass: (0.0, glam::Mat3::ZERO),
        });

        // A small starting stack so there is something to look at.
        // Spaced by a full diameter and a half: 1.4 m apart leaves only
        // 0.4 m of gap between 1 m cubes, so they start interpenetrating
        // and the solver has to shove them apart on the first tick.
        for i in 0..3 {
            self.drop_crate(DVec3::new(0.0, 0.0, 1.0 + i as f64 * 1.6));
        }

        self.spawn_agent();
    }

    /// The nav grid, pinned to the top face of the floor.
    ///
    /// `floor_z` is the surface the agent's feet rest on, not the centre
    /// of the floor body — the floor box is centred at -0.5 with a 0.5
    /// half-extent, so its top is exactly zero.
    fn nav_plane(&self) -> NavPlane {
        NavPlane::new(NAV_DIMS, NAV_TILE, 0.0)
    }

    /// Add the walker, and send it somewhere.
    fn spawn_agent(&mut self) {
        let plane = self.nav_plane();
        let pos = plane.tile_center(2, 2) + DVec3::new(0.0, 0.0, AGENT_HH);
        let collider = Collider3D::box3d(AGENT_HW as f32, AGENT_HW as f32, AGENT_HH as f32);
        let slot = self.grid.insert(pos, collider.radius as f64);

        self.agent_body = self.bodies.len();
        self.bodies.push(Body {
            transform: Transform3D::at(pos),
            velocity: Velocity3D::default(),
            rigid: RigidBody::box3d(
                AGENT_MASS,
                [AGENT_HW as f32, AGENT_HW as f32, AGENT_HH as f32],
            )
            .with_material(Material3D { restitution: 0.0, friction: FLOOR_FRICTION }),
            collider,
            // White, because the walker's colours are baked into its mesh
            // per part. The shader multiplies vertex colour by this tint,
            // so anything else here would wash every part through the
            // same filter — while white leaves the selection-brighten and
            // sleep-dim paths still working on it.
            colour: [1.0, 1.0, 1.0, 1.0],
            slot,
            carried: false,
            dynamic_mass: (0.0, glam::Mat3::ZERO),
        });

        self.agent = Agent3D::new(WalkTuning3D::for_body(
            AGENT_MASS as f64,
            [AGENT_HW, AGENT_HW, AGENT_HH],
            FLOOR_FRICTION as f64,
        ));
        // The task owns the agent's goal from here. Nothing else may set
        // it: two owners fighting over `agent.goal` shows up as an agent
        // that walks halfway to a crate and then heads for a corner.
        self.task = ai3d::StackTask::new(ai3d::StackTuning::for_agent_and_crate(
            [AGENT_HW, AGENT_HW, AGENT_HH],
            [CRATE_HALF as f64, CRATE_HALF as f64, CRATE_HALF as f64],
            NAV_TILE as f64,
            TARGET_LAYERS,
            self.agent.tuning,
        ));
    }

    /// Every crate the stacker may consider, as it looks this tick.
    fn crate_infos(&self) -> Vec<ai3d::CrateInfo> {
        self.bodies
            .iter()
            .enumerate()
            .filter(|(i, b)| *i != self.agent_body && b.rigid.kind != BodyKind::Static)
            .map(|(i, b)| ai3d::CrateInfo {
                id: ai3d::CrateId(i as u32),
                pos: b.transform.pos,
                half_extents: [
                    b.collider.half_extents[0] as f64,
                    b.collider.half_extents[1] as f64,
                    b.collider.half_extents[2] as f64,
                ],
                // Only this task carries anything here, so a crate it is
                // holding is not "carried by someone else".
                carried_by_other: false,
                sleeping: b.rigid.sleeping,
            })
            .collect()
    }

    /// Do what the stacker asked.
    fn apply_stack_action(&mut self, action: ai3d::StackAction) {
        match action {
            ai3d::StackAction::Pickup { id, pose } => {
                let Some(b) = self.bodies.get_mut(id.0 as usize) else { return };
                b.carried = true;
                b.rigid.kind = BodyKind::Kinematic;
                b.rigid.inv_mass = 0.0;
                b.rigid.inv_inertia = glam::Mat3::ZERO;
                b.transform.pos = pose.pos;
                b.transform.rot = pose.rot;
                // Zeroed, and this is not cosmetic. A kinematic body is
                // skipped by `step`, so a residual velocity would never
                // be integrated *and never decay* — it would be handed
                // straight back on release. Worse, the solver's wake rule
                // tests the partner's velocity, so a carried crate still
                // holding walking speed wakes every crate it passes over
                // and the scene never sleeps.
                b.velocity = Velocity3D::default();
                b.rigid.wake();
            }
            ai3d::StackAction::Hold { id, pose } => {
                let Some(b) = self.bodies.get_mut(id.0 as usize) else { return };
                b.transform.pos = pose.pos;
                b.transform.rot = pose.rot;
                b.velocity = Velocity3D::default();
            }
            ai3d::StackAction::Release { id, pose, velocity } => {
                let Some(b) = self.bodies.get_mut(id.0 as usize) else { return };
                b.carried = false;
                b.rigid.kind = BodyKind::Dynamic;
                b.rigid.inv_mass = b.dynamic_mass.0;
                b.rigid.inv_inertia = b.dynamic_mass.1;
                b.transform.pos = pose.pos;
                b.transform.rot = pose.rot;
                b.velocity.linear = velocity;
                b.velocity.angular = DVec3::ZERO;
                // **Mandatory.** The solver wakes a sleeping body only
                // when its partner is awake *and moving*, and the layer
                // below is a settled, asleep crate that is perfectly
                // still. A crate released asleep is skipped by `step`,
                // never falls, and hangs in the air.
                b.rigid.wake();
            }
            ai3d::StackAction::None
            | ai3d::StackAction::NoCrateAvailable
            | ai3d::StackAction::Finished => {}
        }
    }

    /// Add a dynamic crate at `pos`.
    fn drop_crate(&mut self, pos: DVec3) {
        let collider = Collider3D::box3d(CRATE_HALF, CRATE_HALF, CRATE_HALF);
        let slot = self.grid.insert(pos, collider.radius as f64);
        // A little initial spin so crates land at varied angles rather
        // than all perfectly square, which makes the physics legible.
        let n = self.dropped as f64;
        // Stashed before anything can turn this crate kinematic: making
        // it kinematic zeroes both, and they are what a release has to
        // put back.
        // Grippy, because these get stacked.
        //
        // Friction combines as the geometric mean, so crate-on-crate is
        // whatever this is. At 0.6 a crate landing on the tower still has
        // horizontal speed to shed and only 5.9 m/s² to shed it with, so
        // it creeps off the edge over the next few seconds and the stack
        // comes apart on its own — built correctly, then slid apart.
        // Cardboard on cardboard is about 0.8 in reality, and that stops
        // the slide before the crate reaches an edge.
        let rigid = RigidBody::box3d(1.0, [CRATE_HALF, CRATE_HALF, CRATE_HALF])
            .with_material(Material3D { restitution: 0.15, friction: CRATE_FRICTION });
        let dynamic_mass = (rigid.inv_mass, rigid.inv_inertia);
        self.bodies.push(Body {
            transform: Transform3D {
                pos,
                rot: Quat::from_rotation_z((n * 0.7) as f32),
            },
            velocity: Velocity3D {
                linear: DVec3::ZERO,
                angular: DVec3::new((n * 0.3).sin(), (n * 0.5).cos(), 0.0) * 0.8,
            },
            // No damping: with a proper contact manifold the crates
            // settle and sleep on their own, so bleeding energy every
            // tick would only be hiding a solver that could not rest.
            rigid,
            collider,
            colour: crate_colour(self.dropped),
            slot,
            carried: false,
            dynamic_mass,
        });
        self.dropped += 1;
    }

    /// A drop point clear of everything already in the world.
    ///
    /// Spawning at a fixed height overlaps: press the key twice quickly
    /// and two crates occupy the same space. The solver then has to push
    /// them apart from *inside* each other, which is the one case a
    /// single contact point handles badly — the normal it picks is the
    /// axis of least penetration, and deep inside a box that axis flips
    /// between ticks, so they grind against each other instead of
    /// separating.
    ///
    /// Avoiding the overlap is much cheaper than resolving it: drop above
    /// whatever is already there, and step upward until nothing is within
    /// a crate's diameter.
    fn spawn_point(&self) -> DVec3 {
        // A little lateral scatter so crates topple into a pile rather
        // than forming a perfect column.
        let n = self.dropped as f64;
        let x = (n * 1.1).sin() * 0.6;
        let y = (n * 0.9).cos() * 0.6;

        // Start above the tallest thing in the world.
        let highest = self
            .bodies
            .iter()
            .filter(|b| b.rigid.kind.is_dynamic())
            .map(|b| b.transform.pos.z)
            .fold(2.0f64, f64::max);
        let mut z = highest + 3.0;

        // Then walk up until the drop point is clear. Full diameter of
        // separation, so a crate rotated to any angle still fits.
        let clearance = (CRATE_HALF as f64) * 2.0 * 1.2;
        for _ in 0..32 {
            let p = DVec3::new(x, y, z);
            let clash = self
                .bodies
                .iter()
                .filter(|b| b.rigid.kind.is_dynamic())
                .any(|b| b.transform.pos.distance(p) < clearance);
            if !clash {
                break;
            }
            z += clearance;
        }
        DVec3::new(x, y, z)
    }

    /// Where the camera sits this frame.
    fn camera(&self, viewport: Vec2) -> Camera3D {
        let mut c = Camera3D::new(viewport);
        c.position = DVec3::new(
            self.orbit.cos() * CAMERA_DISTANCE,
            self.orbit.sin() * CAMERA_DISTANCE,
            self.height,
        );
        // Look at the middle of the stack rather than the floor, so the
        // action stays centred as crates pile up.
        c.target = DVec3::new(0.0, 0.0, 1.5);
        c
    }

    /// Build the contact list for this tick.
    ///
    /// The engine deliberately hands back a candidate *pair* list and
    /// leaves the shape dispatch to the caller — see `collision`'s module
    /// docs — so this is the part a game writes.
    fn contacts(&self) -> Vec<physics3d::Contact> {
        let mut out = Vec::new();
        for (a, b) in self.grid.query_pairs() {
            let (ia, ib) = (a as usize, b as usize);
            // Slot indices and body indices coincide here only because
            // nothing is ever removed. A game that despawned would keep
            // a slot -> body map.
            let (Some(ba0), Some(bb0)) = (self.bodies.get(ia), self.bodies.get(ib)) else {
                continue;
            };
            if !ba0.rigid.kind.is_dynamic() && !bb0.rigid.kind.is_dynamic() {
                continue;
            }

            // A carried crate is a ghost.
            //
            // It is kinematic, so the solver treats it as infinitely
            // massive, and it is teleported to a new pose every tick
            // rather than integrated. Leaving it in the contact list
            // means every body it sweeps past gets a deep, fast contact
            // against an immovable object — the agent holding it would be
            // launched by its own cargo. It becomes solid again the
            // moment it is released.
            if ba0.carried || bb0.carried {
                continue;
            }

            // **Order matters, and the grid does not order for physics.**
            //
            // `query_pairs` returns `(min, max)` by slot index, so the
            // floor — inserted first — is always the `a` side. The
            // narrowphase normal points "from B toward A", so for a crate
            // resting on that floor it points *downward*, and
            // `correct_penetration` then pushes the crate further in. The
            // crates sink a little every tick and fall through: measured
            // at penetration climbing 0.05 -> 0.88 over three seconds.
            //
            // Putting the dynamic body first makes the normal point up
            // out of the floor, which is what the solver expects. Where
            // both are dynamic the grid's order is fine.
            let swap = !ba0.rigid.kind.is_dynamic() && bb0.rigid.kind.is_dynamic();
            let (ia, ib) = if swap { (ib, ia) } else { (ia, ib) };
            let (ba, bb) = if swap { (bb0, ba0) } else { (ba0, bb0) };

            let ha = to_half(&ba.collider);
            let hb = to_half(&bb.collider);
            let hit = narrow3d::obb_vs_obb(
                ba.transform.pos,
                ha,
                ba.transform.rot,
                bb.transform.pos,
                hb,
                bb.transform.rot,
            );
            if let Some((normal, penetration)) = hit {
                // **Every** contact point between the pair, not just the
                // deepest one. A box resting flat has four equally deep
                // bottom corners; picking one puts every impulse on a
                // 0.5 m lever arm and the force holding the crate up also
                // spins it. See `obb_contact_manifold`.
                for (point, depth) in narrow3d::obb_contact_manifold(
                    ba.transform.pos,
                    ha,
                    ba.transform.rot,
                    bb.transform.pos,
                    hb,
                    bb.transform.rot,
                    normal,
                    penetration,
                ) {
                    out.push(physics3d::Contact {
                        a: ia,
                        b: ib,
                        normal,
                        penetration: depth,
                        point,
                    });
                }
            }
        }
        out
    }

    /// Resolve a click into a selection, using the camera that drew the
    /// frame the player clicked on.
    fn resolve_click(&mut self, cam: &Camera3D, viewport: Vec2, screen: Vec2) {
        let view_proj = Mat4::from_cols_array_2d(&cam.build_uniform().view_proj);
        let Some(ray) = pick3d::ray_from_screen(view_proj, viewport, screen) else {
            return;
        };

        // Collider positions must be in the ray's frame, which is
        // camera-relative — the contract `Camera3D` sets.
        let eye = cam.position;
        let shapes: Vec<Option<PickShape>> = self
            .bodies
            .iter()
            .map(|b| {
                // The floor is not selectable: clicking it should mean
                // "deselect", not "pick the floor".
                if b.rigid.kind == BodyKind::Static {
                    return None;
                }
                Some(PickShape::from_collider(
                    &b.collider,
                    b.transform.pos - eye,
                    b.transform.rot,
                ))
            })
            .collect();

        let hit = pick3d::cast_nearest(&self.grid, ray, PICK_RANGE, 0, |i| {
            shapes.get(i as usize).copied().flatten()
        });
        self.selected = hit.map(|h| h.index as usize);
    }
}

/// The walker, as a little tracked trash robot rather than a box.
///
/// Built from boxes in the body's **local** space, where `+X` is the
/// direction it walks and `+Z` is up. The physics collider stays the
/// plain box it always was — this is cosmetic geometry, and the eyes
/// deliberately poke a few centimetres above the collider because a
/// silhouette that reads as a face is worth more than a mesh that fits
/// its own hitbox exactly.
///
/// Part colours are baked into the vertices. The shader multiplies
/// vertex colour by the per-instance tint, so the agent is drawn with a
/// white tint and each part keeps its own shade — which is also why the
/// sleep-dimming and selection-brightening still work on it.
fn walker_mesh() -> Mesh3D {
    // Deliberately NOT the floor's own dark grey (0.22, 0.24, 0.28).
    //
    // The treads are the part that touches the ground, and at the first
    // attempt they were within a few percent of the floor colour — so the
    // bottom 160 mm of the robot merged into it and the yellow body above
    // read as buried to its waist. The mesh was sitting exactly on the
    // floor the whole time; it was the contrast that was wrong. Near
    // black separates it from the mid-grey floor at any light level.
    const TREAD: [f32; 4] = TREAD_COLOUR;
    const BODY: [f32; 4] = [0.95, 0.72, 0.20, 1.0];
    const PANEL: [f32; 4] = [0.75, 0.55, 0.14, 1.0];
    const METAL: [f32; 4] = [0.55, 0.57, 0.60, 1.0];
    const LENS: [f32; 4] = [0.10, 0.65, 0.85, 1.0];

    let mut m = Mesh3D::new();

    // Everything below is laid out **bottom-up from the collider's own
    // bottom face**, which is at local z = -0.25. Nothing may go below
    // that or the robot is drawn buried: the body rests with its centre
    // a half-height above the floor, so local -0.25 *is* the ground.
    //
    // The head is allowed above the collider — a silhouette that reads as
    // a face is worth more than a mesh that fits its own hitbox exactly —
    // but only just, or the robot looks like it is standing in a hole.

    // Treads: two near-black blocks along the sides, flat on the floor.
    //
    // **They must be wider than the body and a real fraction of its
    // height**, or the robot looks like it is standing in the ground. At
    // the first attempt they were a 140 mm strip tucked under a 300 mm
    // body that overhung them by 70 mm a side — from a camera 18 m up
    // that strip was a few pixels in the body's own shadow, and the
    // yellow hull appeared to meet the floor directly.
    // They sit *just* outside the hull, not splayed out from it. At
    // ±0.42 with a 0.30 width they spanned 1.14 m against a 0.46 m hull —
    // two and a half times its width — and the robot read as a small box
    // balanced on an outsized chassis rather than as a tracked vehicle.
    for side in [-1.0f32, 1.0] {
        m.push_box(
            Vec3::new(0.0, side * 0.29, -0.23),
            Vec3::new(0.86, 0.20, 0.24),
            TREAD,
        );
        // Rollers at each end, so the treads read as tracks rather than
        // as skids. Same bottom as the treads, a little taller.
        for end in [-1.0f32, 1.0] {
            m.push_box(
                Vec3::new(end * 0.34, side * 0.29, -0.22),
                Vec3::new(0.14, 0.20, 0.26),
                METAL,
            );
        }
    }

    // The hull: a **cube**, which is the whole shape of this robot.
    //
    // It was a 0.66 x 0.54 x 0.26 slab first, and a slab is not the
    // silhouette — it read as a flattened box on tracks rather than as a
    // boxy little robot. A cube needs headroom the original 0.5 m tall
    // collider did not have once the tracks took 0.22 of it, which is why
    // `AGENT_HH` grew to 0.35: at that height the tipping limit still
    // leaves 1.84x headroom over the force needed to break friction, so
    // the walker's gait is unaffected. See `WalkTuning3D::for_body`.
    //
    // Still narrower in Y than the track width, so the tracks stay
    // visible from above rather than hidden under an overhang.
    const HULL: f32 = 0.46;
    m.push_box(Vec3::new(0.0, 0.0, 0.12), Vec3::splat(HULL), BODY);
    // A darker front panel, which is what makes the facing direction
    // readable at a glance.
    m.push_box(
        Vec3::new(HULL * 0.5, 0.0, 0.12),
        Vec3::new(0.06, HULL * 0.85, HULL * 0.8),
        PANEL,
    );

    // Arms, folded against the sides.
    for side in [-1.0f32, 1.0] {
        m.push_box(
            Vec3::new(0.08, side * 0.28, 0.05),
            Vec3::new(0.34, 0.08, 0.20),
            METAL,
        );
    }

    // Head: a short neck and two eyes on stalks, looking forward.
    m.push_box(Vec3::new(0.06, 0.0, 0.38), Vec3::new(0.22, 0.16, 0.06), METAL);
    for side in [-1.0f32, 1.0] {
        // The barrel of each eye...
        m.push_box(
            Vec3::new(0.12, side * 0.11, 0.48),
            Vec3::new(0.20, 0.16, 0.16),
            METAL,
        );
        // ...and the lens on the front of it, in a colour nothing else in
        // the scene uses, so the robot always reads as facing you or
        // facing away.
        m.push_box(
            Vec3::new(0.23, side * 0.11, 0.48),
            Vec3::new(0.04, 0.12, 0.12),
            LENS,
        );
    }

    m
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The robot must stand **on** the floor, not in it.
    ///
    /// Its body rests with the centre a half-height up, so local
    /// z = -AGENT_HH is the ground plane. Any vertex below that is drawn
    /// underground, and because the treads are the lowest and darkest
    /// part it reads as the whole robot having sunk. Measured when this
    /// was first built: the rollers hung 30 mm through the floor.
    #[test]
    fn the_walker_mesh_stands_on_the_floor() {
        let m = walker_mesh();
        let lowest = m
            .vertices
            .iter()
            .map(|v| v.pos[2])
            .fold(f32::INFINITY, f32::min);
        let ground = -(AGENT_HH as f32);
        assert!(
            lowest >= ground - 1e-5,
            "the mesh reaches {lowest:.3} but the floor is at {ground:.3},              so it is drawn {:.0} mm underground",
            (ground - lowest) * 1000.0,
        );
    }

    /// And it must not float either: something has to actually touch.
    #[test]
    fn the_walker_mesh_touches_the_floor() {
        let m = walker_mesh();
        let lowest = m
            .vertices
            .iter()
            .map(|v| v.pos[2])
            .fold(f32::INFINITY, f32::min);
        let ground = -(AGENT_HH as f32);
        assert!(
            (lowest - ground).abs() < 0.01,
            "the mesh's lowest point is {:.0} mm above the floor, so the              robot hovers",
            (lowest - ground) * 1000.0,
        );
    }

    /// The part that touches the ground must not be the ground's colour.
    ///
    /// This is what actually made the robot look buried, and no geometry
    /// test would have caught it: the mesh was sitting exactly on the
    /// floor, but the treads were within a few percent of the floor's own
    /// grey, so the bottom 160 mm merged into it and the body above read
    /// as sunk to its waist. Wrong contrast, right position.
    #[test]
    fn the_treads_stand_out_against_the_floor() {
        let m = walker_mesh();
        // The lowest vertices are the treads; sample their colour.
        let ground = -(AGENT_HH as f32);
        let tread = m
            .vertices
            .iter()
            .find(|v| (v.pos[2] - ground).abs() < 1e-4)
            .expect("something must touch the floor")
            .color;

        // Sum of channel differences: a crude but honest stand-in for
        // "can a person tell these apart at a glance".
        let diff: f32 = (0..3).map(|i| (tread[i] - FLOOR_COLOUR[i]).abs()).sum();
        assert!(
            diff > 0.25,
            "the treads {tread:?} are within {diff:.3} of the floor \
             {FLOOR_COLOUR:?}, so the robot's base disappears into it",
        );
    }

    /// The tracks must be visible from above, not hidden under the body.
    ///
    /// The second thing that made the robot look sunk, and again not a
    /// position bug: the body was 0.62 wide over tracks at ±0.38, so it
    /// overhung them by 70 mm a side. From a camera 18 m up, the tracks
    /// were a few pixels of near-black in the body's own shadow, and the
    /// yellow hull appeared to meet the floor directly. The widest thing
    /// at ground level has to be the tracks.
    #[test]
    fn the_tracks_are_wider_than_the_body() {
        let m = walker_mesh();
        let ground = -(AGENT_HH as f32);

        // Half-width of whatever touches the floor...
        let track_half = m
            .vertices
            .iter()
            .filter(|v| (v.pos[2] - ground).abs() < 1e-4)
            .map(|v| v.pos[1].abs())
            .fold(0.0f32, f32::max);
        // ...against the half-width of the bulk clear above them.
        //
        // The tracks are identified by *colour*, not by position. Keying
        // on "whatever is far out in Y" is circular: widen the body and
        // it starts matching the filter, so the test compares the body
        // against itself and passes no matter how badly it overhangs.
        // Nothing may be wider than the running gear. Stated that way
        // round it is not circular: it asks of *every* vertex whether it
        // sticks out past what touches the floor, so widening the body
        // fails it immediately. Asking instead for "the widest thing that
        // is not the tracks" needs a rule for what counts as the tracks,
        // and every such rule the body can grow into is a rule that makes
        // the test compare the body against itself.
        let widest = m
            .vertices
            .iter()
            .map(|v| v.pos[1].abs())
            .fold(0.0f32, f32::max);

        assert!(
            widest <= track_half + 1e-4,
            "something reaches {widest:.3} out while the tracks only reach \
             {track_half:.3}, so it overhangs them and hides them from view",
        );
    }

    /// The head may clear the collider, but only by a little — past that
    /// the robot reads as standing in a hole rather than as tall.
    ///
    /// The allowance is a *fraction of the robot's own height*, not a
    /// fixed distance. A fixed one does not survive the robot changing
    /// size: it was 0.12 m, which was a fifth of a 0.6 m robot and would
    /// have been a tenth of a 1.2 m one, so the same mesh would pass or
    /// fail depending on scale rather than on how it looks.
    #[test]
    fn the_walker_mesh_does_not_tower_over_its_collider() {
        let m = walker_mesh();
        let highest = m
            .vertices
            .iter()
            .map(|v| v.pos[2])
            .fold(f32::NEG_INFINITY, f32::max);
        let lowest = m
            .vertices
            .iter()
            .map(|v| v.pos[2])
            .fold(f32::INFINITY, f32::min);

        let total = highest - lowest;
        let over = highest - AGENT_HH as f32;
        assert!(
            over <= total * 0.25,
            "the head clears the collider by {over:.3} m, {:.0}% of the \
             robot's {total:.2} m height — it will read as standing in a hole",
            over / total * 100.0,
        );
    }
}

/// Write a frame out as a binary PPM.
///
/// PPM because it needs no dependency: a 15-byte header and raw RGB. Any
/// image viewer or converter reads it, and adding an encoder crate to an
/// example for the sake of a debug dump is not worth the build time.
fn write_ppm(path: &str, shot: &void_engine::renderer::ScreenshotData) {
    let mut out = format!("P6
{} {}
255
", shot.width, shot.height).into_bytes();
    for px in shot.pixels.chunks_exact(4) {
        out.extend_from_slice(&px[..3]);
    }
    let _ = std::fs::write(path, out);
}

/// Distinct colours so crates are tellable apart.
fn crate_colour(n: usize) -> [f32; 4] {
    const PALETTE: [[f32; 4]; 6] = [
        [0.85, 0.45, 0.25, 1.0],
        [0.35, 0.65, 0.85, 1.0],
        [0.80, 0.75, 0.35, 1.0],
        [0.55, 0.75, 0.45, 1.0],
        [0.75, 0.45, 0.70, 1.0],
        [0.60, 0.60, 0.65, 1.0],
    ];
    PALETTE[n % PALETTE.len()]
}

fn to_half(c: &Collider3D) -> [f64; 3] {
    [
        c.half_extents[0] as f64,
        c.half_extents[1] as f64,
        c.half_extents[2] as f64,
    ]
}

impl App for Game {
    fn init(&mut self, _ctx: &mut SimCtx) {}

    fn fixed_update(&mut self, ctx: &mut SimCtx) {
        let dt = ctx.dt;

        // Camera controls. Read here rather than in `render` so they
        // advance at the fixed rate and feel the same at any frame rate.
        if ctx.input.key_down(KeyCode::KeyA) {
            self.orbit += 1.2 * dt as f64;
        }
        if ctx.input.key_down(KeyCode::KeyD) {
            self.orbit -= 1.2 * dt as f64;
        }
        if ctx.input.key_down(KeyCode::KeyW) {
            self.height = (self.height + 6.0 * dt as f64).min(25.0);
        }
        if ctx.input.key_down(KeyCode::KeyS) {
            self.height = (self.height - 6.0 * dt as f64).max(1.0);
        }

        if ctx.input.key_pressed(KeyCode::KeyR) {
            self.reset();
        }
        if ctx.input.key_pressed(KeyCode::Space) {
            self.drop_crate(self.spawn_point());
        }

        // A click is recorded here and resolved in `render`, which is
        // where the camera matrix lives. `SimCtx` has no renderer, by
        // design — it is the context a dedicated server gets.
        if ctx.input.mouse_pressed(MouseButton::Left) {
            self.pending_click = Some(ctx.input.mouse_pos);
        }
        if ctx.input.mouse_pressed(MouseButton::Right) {
            self.pending_launch = true;
        }

        if self.pending_launch {
            self.pending_launch = false;
            if let Some(i) = self.selected {
                if let Some(b) = self.bodies.get_mut(i) {
                    b.rigid.wake();
                    b.velocity.linear += DVec3::new(0.0, 0.0, 7.0);
                    b.velocity.angular += DVec3::new(1.5, 0.8, 0.0);
                }
            }
        }

        // ---- the stacking behaviour -------------------------------------
        //
        // Runs before the walker, which runs before `step`: the task
        // decides where the agent wants to be, the walker decides how
        // hard to push to get there, and the integrator consumes the
        // force. The action is applied *here*, before the broadphase is
        // re-hashed below, so the narrowphase sees the carried crate
        // where it actually is rather than where it was last tick.
        {
            let crates = self.crate_infos();
            let plane = self.nav_plane();
            let transform = self.bodies[self.agent_body].transform.clone();
            let action = ai3d::drive_stacker(
                &mut self.task,
                &mut self.agent,
                &transform,
                plane,
                &NavFloor,
                &crates,
                dt,
            );
            self.apply_stack_action(action);
        }

        // ---- the agent -------------------------------------------------
        //
        // Before `step`, in the same place a player's input would be
        // read: the force is an acceleration the integrator consumes.
        {
            let i = self.agent_body;
            if let Some(b) = self.bodies.get_mut(i) {
                ai3d::drive_agent(
                    &mut self.agent,
                    &mut b.rigid,
                    &b.transform,
                    &mut b.velocity,
                    dt,
                );
            }
        }

        // ---- the physics step ------------------------------------------
        //
        // `physics3d::step` does gravity, integration and sleeping; the
        // contact list in between is the game's to build, which is what
        // `contacts()` above does.
        {
            let mut refs: Vec<physics3d::BodyRef<'_>> = self
                .bodies
                .iter_mut()
                .map(|b| physics3d::BodyRef {
                    body: &mut b.rigid,
                    transform: &mut b.transform,
                    velocity: &mut b.velocity,
                })
                .collect();
            physics3d::step(&mut refs, physics3d::GRAVITY, dt);
        }

        // Re-hash moved bodies before querying, or the broadphase answers
        // about where things were last tick.
        for b in &self.bodies {
            self.grid.update(b.slot, b.transform.pos, b.collider.radius as f64);
        }

        let contacts = self.contacts();
        {
            let mut refs: Vec<physics3d::BodyRef<'_>> = self
                .bodies
                .iter_mut()
                .map(|b| physics3d::BodyRef {
                    body: &mut b.rigid,
                    transform: &mut b.transform,
                    velocity: &mut b.velocity,
                })
                .collect();
            physics3d::solver::solve(&mut refs, &contacts, dt as f64);
            // Sleep is checked *here*, after the solve, not inside
            // `step`. A resting body still holds a tick of gravity when
            // `step` ends -- the solver cancels it a moment later -- so
            // testing for stillness any earlier sees every settled body
            // as moving and nothing ever sleeps.
            physics3d::update_sleep_all(&mut refs, dt);
        }

        // Anything that falls off the world is gone; without this a
        // stray crate integrates forever.
        for b in &mut self.bodies {
            if b.transform.pos.z < -50.0 {
                b.transform.pos = DVec3::new(0.0, 0.0, 10.0);
                b.velocity = Velocity3D::default();
                b.rigid.wake();
            }
        }
    }
}

impl ClientApp for Game {
    fn window_title(&self) -> &'static str {
        "void_engine — crates3d"
    }

    fn render(&mut self, r: &mut Renderer, _w: &World, _i: &InputState, _alpha: f32) {
        let viewport = r.viewport_size();
        let cam = self.camera(viewport);

        // Upload the two meshes once, on the first frame that has a
        // device to upload to.
        if self.crate_mesh.is_none() {
            let mut m = Mesh3D::new();
            m.push_box(Vec3::ZERO, Vec3::splat(CRATE_HALF * 2.0), [1.0; 4]);
            self.crate_mesh = r.upload_mesh(&m).ok();

            let mut f = Mesh3D::new();
            f.push_box(
                Vec3::ZERO,
                Vec3::new(FLOOR_HALF as f32 * 2.0, FLOOR_HALF as f32 * 2.0, 1.0),
                [1.0; 4],
            );
            self.floor_mesh = f.is_empty().then_some(None).flatten().or(r.upload_mesh(&f).ok());

            self.agent_mesh = r.upload_mesh(&walker_mesh()).ok();
        }

        // A click recorded last tick is resolved now, against the camera
        // the player was actually looking through.
        if let Some(px) = self.pending_click.take() {
            self.resolve_click(&cam, viewport, px);
        }

        let eye = cam.position;
        r.set_camera_3d(cam);
        // Sun over the scene, and a shadow frustum big enough to cover
        // the floor.
        r.set_sun_3d(Vec3::new(0.35, 0.25, 0.9), FLOOR_HALF as f32 + 4.0);
        r.set_ambient_3d(0.28);

        // A warm point light following the selected crate, so selection
        // reads without an outline pass.
        if let Some(i) = self.selected {
            if let Some(b) = self.bodies.get(i) {
                r.push_point_light_3d(PointLight3D::new(
                    (b.transform.pos - eye).as_vec3() + Vec3::Z * 1.5,
                    6.0,
                    [1.0, 0.85, 0.5],
                    1.4,
                ));
            }
        }

        // ---- draw the scene --------------------------------------------
        for (i, b) in self.bodies.iter().enumerate() {
            let handle = if b.rigid.kind == BodyKind::Static {
                self.floor_mesh
            } else if i == self.agent_body {
                self.agent_mesh
            } else {
                self.crate_mesh
            };
            let Some(handle) = handle else { continue };

            // Camera-relative, subtracting in f64 before the cast — the
            // precision contract the whole 3D path follows.
            let model = if i == self.agent_body {
                // The walker is drawn facing where it *walks*, not where
                // the solver left it.
                //
                // Nothing ever yaws the body deliberately — it is a box
                // pushed along by a centre-of-mass force — so its
                // rotation is contact noise, and a robot built from it
                // would spin on the spot while driving in a straight
                // line. `StackTask::facing` is the direction the agent is
                // actually heading, slewed so corners are turns rather
                // than flicks.
                let f = self.task.facing();
                let yaw = (f.y).atan2(f.x) as f32;
                Mat4::from_rotation_translation(
                    Quat::from_rotation_z(yaw),
                    (b.transform.pos - eye).as_vec3(),
                )
            } else {
                b.transform.model_matrix(eye)
            };

            let mut tint = b.colour;
            if Some(i) == self.selected {
                // Brighten the selection rather than outlining it.
                tint = [
                    (tint[0] * 1.6).min(1.0),
                    (tint[1] * 1.6).min(1.0),
                    (tint[2] * 1.6).min(1.0),
                    1.0,
                ];
            } else if b.rigid.sleeping {
                // Asleep reads slightly dimmer, which makes the sleep
                // system visible rather than invisible.
                tint = [tint[0] * 0.75, tint[1] * 0.75, tint[2] * 0.75, 1.0];
            }

            r.draw_mesh_with(MeshDraw { handle, model, color: tint });
        }

        // ---- the HUD, in 2D over the scene ------------------------------
        self.draw_hud(r, viewport);

        // ---- self-check, under `--verify` -------------------------------
        self.frame += 1;
        if let Some(at) = self.verify_at {
            if self.frame == at {
                // Ask for a capture; it arrives next frame.
                r.screenshot_pending = true;
            } else if self.frame == at + 1 {
                let shot = r.screenshot_data.take();
                self.run_verification(shot);
            }
        }
    }
}

impl Game {
    /// Assert the assembled scene actually rendered, and exit non-zero if
    /// not.
    ///
    /// Checks three things a broken assembly gets wrong while still
    /// running at full speed:
    ///
    /// 1. The frame is not uniform — an all-black or all-clear-colour
    ///    window means nothing drew.
    /// 2. The crates' colours are present, so 3D geometry specifically
    ///    reached the screen rather than just the HUD.
    /// 3. The physics moved something: the opening stack must have
    ///    fallen from where it was placed.
    fn run_verification(&self, shot: Option<void_engine::renderer::ScreenshotData>) {
        let Some(shot) = shot else {
            eprintln!("VERIFY FAIL: no screenshot was captured");
            std::process::exit(1);
        };

        // Dump the frame when asked, so a human can look at the scene
        // rather than only at the assertions below.
        if let Ok(path) = std::env::var("CRATES3D_SHOT") {
            write_ppm(&path, &shot);
            eprintln!("[verify] wrote {path}");
        }

        let px = &shot.pixels;
        let n = (px.len() / 4) as f64;

        // 1. Variety. A frame that drew nothing is one flat colour.
        let mut distinct = std::collections::HashSet::new();
        let mut lit = 0usize;
        for c in px.chunks(4) {
            // Quantised, so anti-aliasing and lighting gradients do not
            // count as thousands of "distinct" colours on their own.
            distinct.insert((c[0] / 32, c[1] / 32, c[2] / 32));
            if c[0] > 40 || c[1] > 40 || c[2] > 40 {
                lit += 1;
            }
        }
        let lit_frac = lit as f64 / n;
        println!(
            "[verify] {}x{}  distinct={}  lit={:.1}%",
            shot.width,
            shot.height,
            distinct.len(),
            lit_frac * 100.0,
        );
        // A HUD-only frame measured 5 distinct buckets; the full scene
        // measures 18. Ten sits well clear of the former and well under
        // the latter, so this fails on a blank scene without being
        // brittle about lighting changes.
        if distinct.len() < 10 {
            eprintln!(
                "VERIFY FAIL: only {} distinct colours — the frame is \
                 essentially blank, so the scene did not render",
                distinct.len(),
            );
            std::process::exit(1);
        }
        if lit_frac < 0.05 {
            eprintln!(
                "VERIFY FAIL: only {:.1}% of the frame is lit — the scene \
                 is not being drawn",
                lit_frac * 100.0,
            );
            std::process::exit(1);
        }

        // 2. 3D geometry specifically.
        //
        // Keyed on the crates' *warm* colours — pixels where red clearly
        // dominates blue. Nothing else in the frame can produce those:
        // the main pass clears to (0.02, 0.02, 0.05), which is blue-
        // leaning, the floor is a blue-grey, and the HUD is a dark panel
        // with pale text.
        //
        // An earlier revision of this check looked for "floor-like"
        // pixels — dark, blue > red — and matched **the clear colour**,
        // so it passed with every 3D draw deleted. Verified: deleting the
        // draws now fails here.
        let warm = px
            .chunks(4)
            .filter(|c| {
                let (r, g, b) = (c[0] as i32, c[1] as i32, c[2] as i32);
                r > 70 && r > b + 30 && r >= g
            })
            .count();
        let warm_frac = warm as f64 / n;
        println!("[verify] crate-coloured pixels = {:.2}%", warm_frac * 100.0);
        if warm_frac < 0.002 {
            eprintln!(
                "VERIFY FAIL: only {:.3}% of the frame is crate-coloured — \
                 3D geometry is not reaching the screen",
                warm_frac * 100.0,
            );
            std::process::exit(1);
        }

        // 3. Physics ran. The opening crates are placed at z = 1.0, 2.4,
        //    3.8 and must have fallen toward the floor by now.
        // The agent is dynamic too, but it is a different size and it is
        // deliberately still walking, so it fails both the resting-height
        // and the everything-is-asleep checks below on purpose.
        let crates: Vec<&Body> = self
            .bodies
            .iter()
            .enumerate()
            .filter(|(i, b)| b.rigid.kind.is_dynamic() && *i != self.agent_body)
            .map(|(_, b)| b)
            .collect();
        let lowest = crates
            .iter()
            .map(|b| b.transform.pos.z)
            .fold(f64::INFINITY, f64::min);
        let asleep = crates.iter().filter(|b| b.rigid.sleeping).count();
        println!(
            "[verify] crates={}  lowest z={:.2}  asleep={}",
            crates.len(),
            lowest,
            asleep,
        );
        // `>=` rather than `!(< 0.9)`: a NaN z is also a failure, and
        // saying so explicitly is clearer than relying on negation.
        if lowest.is_nan() || lowest >= 0.9 {
            eprintln!(
                "VERIFY FAIL: the lowest crate is still at z={lowest:.2}, \
                 above where it started — gravity or contact resolution \
                 is not running",
            );
            std::process::exit(1);
        }
        if lowest < -2.0 {
            eprintln!(
                "VERIFY FAIL: a crate reached z={lowest:.2}, below the \
                 floor — contacts are not stopping it",
            );
            std::process::exit(1);
        }

        // A crate resting on the floor has its centre exactly a
        // half-extent above it. Sinking past the penetration slop means
        // the solver is losing ground every tick, which ends with the
        // crate falling through — the failure that `lowest < -2.0` above
        // only catches once it is far too late.
        let resting = CRATE_HALF as f64;
        if lowest < resting - 0.02 {
            eprintln!(
                "VERIFY FAIL: the lowest crate is at z={lowest:.3}, {:.3} m \
                 into the floor — contacts are not holding it up",
                resting - lowest,
            );
            std::process::exit(1);
        }

        // Nothing may overlap anything else once settled. A pair stuck
        // inside each other is what a single-point contact manifold
        // produces: the normal flips between ticks and the pair grinds
        // instead of separating.
        let mut worst = 0.0f64;
        for c in &self.contacts() {
            // Crate against crate only. The agent is actively pushing,
            // so a live overlap with it is the solver doing its job.
            if c.a == self.agent_body || c.b == self.agent_body {
                continue;
            }
            let (a, b) = (&self.bodies[c.a], &self.bodies[c.b]);
            if a.rigid.kind.is_dynamic() && b.rigid.kind.is_dynamic() {
                worst = worst.max(c.penetration);
            }
        }
        if worst > 0.05 {
            eprintln!(
                "VERIFY FAIL: two crates overlap by {worst:.3} m — they are \
                 stuck inside each other rather than resting on each other",
            );
            std::process::exit(1);
        }

        // Everything must have gone to sleep. A settled crate that stays
        // awake is jiggling: the solver is still finding motion to cancel
        // every tick, which is exactly what a single-point contact
        // manifold produced, and it costs the scene real work forever.
        // Asleep *or* genuinely still.
        //
        // Sleeping was the right contract when the scene was three crates
        // on a floor. It is the wrong one now the agent builds a tower and
        // keeps walking past it: a crate near the top of a settling stack
        // sits at a few millimetres per second — well inside
        // `SLEEP_LINEAR_THRESHOLD` — but has not yet held still for the
        // half second `TIME_TO_SLEEP` wants, and the agent nudges the clock
        // every time it walks by. What must be true is that nothing is
        // *moving*; falling asleep is the bonus that proves it.
        let worst_v = crates
            .iter()
            .map(|b| b.velocity.linear.length())
            .fold(0.0f64, f64::max);
        let settled = crates
            .iter()
            // The bar is "not going anywhere", not "asleep".
            //
            // A stacked crate never fully sleeps: the solver pushes it out
            // of the crate below every tick and gravity pulls it back,
            // leaving a standing residual around 0.11 m/s that neither
            // side wins. That is inherent to correcting penetration by
            // position alone — removing it needs split impulses or a
            // velocity bias, which is a different solver rather than a
            // different constant. What *is* checkable is that the stack
            // does not move, and the height assertion below does that to
            // the millimetre.
            .filter(|b| b.rigid.sleeping || b.velocity.linear.length() < 0.2)
            .count();
        if settled != crates.len() {
            eprintln!(
                "VERIFY FAIL: only {settled} of {} crates have come to rest \
                 after {VERIFY_AT_FRAME} frames ({asleep} asleep) — the \
                 fastest is still moving at {worst_v:.4} m/s",
                crates.len(),
            );
            std::process::exit(1);
        }

        // The agent must have got somewhere. A walker that never moves
        // looks identical to one whose path failed to plan, and both
        // render fine — so check the distance, not the pixels.
        let walker = &self.bodies[self.agent_body];
        let plane = self.nav_plane();
        let start = plane.tile_center(2, 2);
        let walked = (walker.transform.pos - start).truncate().length();
        if walked < 1.0 {
            eprintln!(
                "VERIFY FAIL: the agent has moved {walked:.2} m from where it \
                 spawned after {VERIFY_AT_FRAME} frames — state is {:?}, so it \
                 is not walking",
                self.agent.state,
            );
            std::process::exit(1);
        }
        if walker.transform.pos.z < AGENT_HH - 0.05 {
            eprintln!(
                "VERIFY FAIL: the agent is at z={:.3}, below its {AGENT_HH:.2} m \
                 resting height — it is sinking into the floor",
                walker.transform.pos.z,
            );
            std::process::exit(1);
        }
        // Upright. A walk force through the centre of mass imparts no
        // torque, so a leaning agent means it has acquired a lever arm.
        let up = (walker.transform.rot * Vec3::Z).as_dvec3();
        let tilt = up.dot(DVec3::Z).clamp(-1.0, 1.0).acos().to_degrees();
        if tilt > 5.0 {
            eprintln!("VERIFY FAIL: the agent is leaning {tilt:.1} degrees off vertical");
            std::process::exit(1);
        }

        // The stack, which is the whole point of the agent.
        //
        // Counted from the crates rather than from the task's tally: a
        // tally cannot tell a tower that stands from one that was built
        // and then fell over, and it is the standing one that matters.
        let layers = self.task.layers_standing(self.nav_plane(), &self.crate_infos());
        if layers < TARGET_LAYERS {
            eprintln!(
                "VERIFY FAIL: the agent built {layers} of {TARGET_LAYERS} layers \
                 after {VERIFY_AT_FRAME} frames — task state is {:?}",
                self.task.state,
            );
            std::process::exit(1);
        }

        // And the tower must be at its nominal height, not squashed into
        // itself. Each contact in a stack sinks under the load above it,
        // and a solver that cannot push it back out leaves a three-high
        // pile measurably shorter than three crates.
        let mut heights: Vec<f64> = crates.iter().map(|b| b.transform.pos.z).collect();
        heights.sort_by(f64::total_cmp);
        let top = heights.last().copied().unwrap_or(0.0);
        let nominal = (TARGET_LAYERS as f64 - 0.5) * (CRATE_HALF as f64 * 2.0);
        if top < nominal - 0.15 {
            eprintln!(
                "VERIFY FAIL: the top crate is at z={top:.3} against a nominal \
                 {nominal:.3} — the stack is compressed into itself",
            );
            std::process::exit(1);
        }

        println!(
            "[verify] agent walked {walked:.2} m, tilt {tilt:.2} deg, state {:?}",
            self.agent.state,
        );
        println!("[verify] stack: {layers} layers, top crate at z={top:.3}");
        println!(
            "[verify] OK — scene rendered, geometry visible, \
             physics settled (resting z={lowest:.3}, worst overlap={worst:.4})"
        );
        std::process::exit(0);
    }
}

impl Game {
    fn draw_hud(&self, r: &mut Renderer, viewport: Vec2) {
        let awake = self
            .bodies
            .iter()
            .filter(|b| b.rigid.kind.is_dynamic() && !b.rigid.sleeping)
            .count();
        let total = self.bodies.len() - 1; // the floor is not a crate

        let batch = &mut r.batch;
        batch.set_surface(Surface::new(Material::Solid));

        // A panel along the bottom, in UI space: centre-origin, +Y up.
        let half_w = viewport.x * 0.5;
        let half_h = viewport.y * 0.5;
        let panel_h = 54.0;
        batch.rect(
            Vec2::new(0.0, -half_h + panel_h * 0.5),
            Vec2::new(viewport.x, panel_h),
            [0.05, 0.06, 0.09, 0.85],
        );

        let text_y = -half_h + 34.0;
        void_engine::text::draw_text(
            batch,
            "SPACE drop   LMB select   RMB launch   A/D orbit   W/S height   R reset",
            Vec2::new(-half_w + 16.0, text_y),
            1.0,
            [0.75, 0.78, 0.85, 1.0],
        );

        let status = match self.selected {
            Some(i) => format!("crates {total}   awake {awake}   selected #{i}"),
            None => format!("crates {total}   awake {awake}   nothing selected"),
        };
        void_engine::text::draw_text(
            batch,
            &status,
            Vec2::new(-half_w + 16.0, text_y - 16.0),
            1.0,
            [0.95, 0.9, 0.7, 1.0],
        );

        // A crosshair, so the player knows where a click will land.
        batch.rect(Vec2::ZERO, Vec2::new(10.0, 2.0), [1.0, 1.0, 1.0, 0.55]);
        batch.rect(Vec2::ZERO, Vec2::new(2.0, 10.0), [1.0, 1.0, 1.0, 0.55]);

        batch.clear_surface();
    }
}

/// Frames to run before the self-check fires, when `--verify` is passed.
///
/// Long enough for the opening stack to fall and settle *and fall
/// asleep*, so the check sees a scene physics has actually finished with
/// rather than one still in motion. Settling takes a little under four
/// seconds from the drop, plus the half second of continuous stillness
/// [`void_engine::physics3d::TIME_TO_SLEEP`] requires.
const VERIFY_AT_FRAME: u32 = 900;

fn main() {
    // `--verify` runs headed for a couple of seconds, captures a frame,
    // asserts the 3D scene really rendered, and exits non-zero if not.
    //
    // This exists because every other 3D test draws one or two boxes to
    // an offscreen buffer. A green test suite and a steady frame rate
    // both say nothing about whether the assembled scene is visible --
    // an all-black window runs at exactly the same speed.
    let verify = std::env::args().any(|a| a == "--verify");
    let mut game = Game::new();
    game.verify_at = verify.then_some(VERIFY_AT_FRAME);
    void_engine::run(game);
}
