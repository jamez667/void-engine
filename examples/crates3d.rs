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
//!
//! # Controls
//!
//! * **Left click** — select the crate under the cursor.
//! * **Right click** — launch the selected crate upward.
//! * **Space** — drop a new crate.
//! * **A / D** — orbit the camera. **W / S** — raise and lower it.
//! * **R** — reset the scene.
//!
//! Run with:
//!     cargo run --release --features render3d --example crates3d

use glam::{DVec3, Mat4, Quat, Vec2, Vec3};
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
use void_engine::{App, ClientApp, SimCtx, World};

/// Half-extent of a crate, in metres.
const CRATE_HALF: f32 = 0.5;
/// Half-extent of the floor.
const FLOOR_HALF: f64 = 12.0;
/// How far the camera orbits from the origin.
const CAMERA_DISTANCE: f64 = 18.0;
/// How far a click reaches.
const PICK_RANGE: f64 = 100.0;

/// One thing in the world: a pose, a body, and how to draw it.
struct Body {
    transform: Transform3D,
    velocity: Velocity3D,
    rigid: RigidBody,
    collider: Collider3D,
    colour: [f32; 4],
    /// Its slot in the broadphase grid, so a pick result maps back here.
    slot: u32,
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
            colour: [0.22, 0.24, 0.28, 1.0],
            slot,
        });

        // A small starting stack so there is something to look at.
        for i in 0..3 {
            self.drop_crate(DVec3::new(0.0, 0.0, 1.0 + i as f64 * 1.4));
        }
    }

    /// Add a dynamic crate at `pos`.
    fn drop_crate(&mut self, pos: DVec3) {
        let collider = Collider3D::box3d(CRATE_HALF, CRATE_HALF, CRATE_HALF);
        let slot = self.grid.insert(pos, collider.radius as f64);
        // A little initial spin so crates land at varied angles rather
        // than all perfectly square, which makes the physics legible.
        let n = self.dropped as f64;
        self.bodies.push(Body {
            transform: Transform3D {
                pos,
                rot: Quat::from_rotation_z((n * 0.7) as f32),
            },
            velocity: Velocity3D {
                linear: DVec3::ZERO,
                angular: DVec3::new((n * 0.3).sin(), (n * 0.5).cos(), 0.0) * 0.8,
            },
            rigid: RigidBody::box3d(
                1.0,
                [CRATE_HALF, CRATE_HALF, CRATE_HALF],
            )
            .with_material(Material3D { restitution: 0.15, friction: 0.6 }),
            collider,
            colour: crate_colour(self.dropped),
            slot,
        });
        self.dropped += 1;
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
                // Contact point approximated as the midpoint between
                // centres, pulled onto the surface. Good enough for boxes
                // this size; a real manifold would give up to four points
                // per face pair and make stacks steadier.
                // The deepest point of A into B. This must be a real
                // surface point, not an approximation: the lever arm from
                // each centre to the contact decides how much of the
                // impulse becomes spin, and a point metres off the
                // surface shrinks the impulse until bodies sink through
                // each other. See `obb_contact_point`.
                let point = narrow3d::obb_contact_point(
                    ba.transform.pos,
                    ha,
                    ba.transform.rot,
                    normal,
                    penetration,
                );
                out.push(physics3d::Contact {
                    a: ia,
                    b: ib,
                    normal,
                    penetration,
                    point,
                });
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
            // Drop from above the current pile, offset so it topples
            // rather than landing perfectly stacked.
            let n = self.dropped as f64;
            self.drop_crate(DVec3::new(
                (n * 1.1).sin() * 0.6,
                (n * 0.9).cos() * 0.6,
                8.0,
            ));
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
            } else {
                self.crate_mesh
            };
            let Some(handle) = handle else { continue };

            // Camera-relative, subtracting in f64 before the cast — the
            // precision contract the whole 3D path follows.
            let model = b.transform.model_matrix(eye);

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
        if std::env::var("CRATES3D_TRACE").is_ok() && self.frame.is_multiple_of(30) {
            let zs: Vec<String> = self
                .bodies
                .iter()
                .filter(|b| b.rigid.kind.is_dynamic())
                .map(|b| format!("{:.3}", b.transform.pos.z))
                .collect();
            let cs = self.contacts();
            let pen = cs.iter().map(|c| c.penetration).fold(0.0f64, f64::max);
            let pairs = self.grid.query_pairs().len();
            let norms: Vec<String> = cs.iter().take(3)
                .map(|c| format!("({},{}) n=({:.2},{:.2},{:.2}) p={:.3}",
                     c.a, c.b, c.normal.x, c.normal.y, c.normal.z, c.penetration)).collect();
            eprintln!("[trace] f{:4} z=[{}] pairs={} contacts={} max_pen={:.4} {}",
                self.frame, zs.join(" "), pairs, cs.len(), pen, norms.join(" | "));
        }
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
        let crates: Vec<&Body> = self
            .bodies
            .iter()
            .filter(|b| b.rigid.kind.is_dynamic())
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

        println!("[verify] OK — scene rendered, geometry visible, physics settled");
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
/// Long enough for the opening stack to fall and settle, so the check
/// sees a scene physics has actually acted on rather than its initial
/// placement.
const VERIFY_AT_FRAME: u32 = 120;

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
