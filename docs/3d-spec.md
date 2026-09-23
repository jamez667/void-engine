# Adding 3D to void_engine — scoping spec

**Status:** proposal, not scheduled. Written 2026-09-21 against `c9b92c4`.

All figures below were measured or read out of the tree at that commit, not
estimated from the README. Where a number contradicts an existing TODO entry
it is flagged.

---

## 1. The headline

The engine is 2D in three separable ways, and they cost wildly different
amounts:

| Layer | 2D-ness | Cost to make 3D |
| --- | --- | --- |
| **Math / netcode / persistence / ECS / app loop** | Shallow or none. Types are named `*2D`, `Vec3`/`Mat4`/`Quat` already re-exported, encoders generic, `SimCtx` projection-free. | **Small.** Additive — new types beside old. |
| **Batch + main shader** | Medium. One `mat4x4` multiply already; vertex `pos` is `[f32;2]`. | **Medium.** Contained, but 408 downstream call sites constrain the API. |
| **Collision / pathfind / terrain noise** | Algorithmic, not just typed: 2D grid-DDA, 4-axis SAT, `(i32,i32)` keys, 4-corner bilerp. | **Medium-large.** Real rewrites, but well-understood and self-contained. |
| **The effect chain** (shadow, lights, godray, sun) | **Total.** Every technique is a screen-space march over one flat binary `wall_mask`. No height data exists to extend. | **Large — replacement, not a port.** |

**The single most important finding:** the cost is *not* spread evenly, and it
is not dominated by the 778 `Vec2` call sites. It is dominated by the
effect chain, which cannot be ported at all — it must be replaced with
different techniques. Everything else is tractable.

**Recommendation: do not convert the engine. Add a parallel 3D path.**
Rationale in §7.

---

## 2. What is already 3D-shaped (the footholds)

These are real and reduce the work materially:

- **`camera.rs:47`** already builds a `Mat4` and uploads it as a
  `view_proj` uniform. It happens to be `Mat4::orthographic_rh`, but the
  *plumbing* — uniform struct, bind group, upload — is projection-agnostic.
  Swapping in `Mat4::perspective_rh` is a few lines.
- **`shader.wgsl:49`** is `camera.view_proj * vec4<f32>(in.position, 0.0, 1.0)`.
  A real 3D vertex path is this line with `in.position` widened and the
  hardcoded `0.0` removed.
- **`math.rs:1`** already re-exports `Vec3, Vec4, Mat4, Quat`. No new
  dependency; glam is already pulling its 3D types in.
- **Components are already namespaced**: `Transform2D`, `Destructible2D`.
  A `Transform3D` lands beside them without renaming a single existing
  field. The registry names (`"transform2d"`, …) are documented as engine API
  that "may not change without a `rename`" — the `2D` suffix keeps that
  promise intact.
- **The wire encoder is per-axis and generic.** `snapshot.rs:275-279`
  calls `write_quantised` once per component. A Z axis is two added lines,
  not a format redesign.
- **A real 3D lighting calculation already exists**, in `terrain::field::hillshade`
  (`terrain/field.rs:106-113`): it builds a surface normal
  `Vec3::new(-g.x, -g.y, 1.0).normalize()` and does a Lambertian `n.dot(l)`
  against a light lifted into z. It is CPU-side over a 2.5D heightfield, but
  it is a genuine normal-based shading model and it establishes the "+Z is up"
  convention.
- **`ShadowPass::write_mask_camera`** (`shadow.rs:354-365`) already drives a
  *second, different* projection through the *same* `camera_bgl` layout —
  exactly the shape a shadow-map light-space matrix needs.
- **Terrain is already a 2.5D heightfield** (`Vec2 -> f32`), though entirely
  disconnected from the simulation: nothing in `physics`, `collision` or
  `components` reads a height, and `Transform2D` has no elevation field.

## 3. What has no 3D equivalent (the real cost)

`frame.rs` is a hardcoded linear sequence of **11 render passes** (there is no
frame-graph abstraction; the pass order is literal code in a 764-line
function). Named at `frame.rs:269`–`534`:

`offscreen → blur_h → blur_v → wall_mask → shadow_raycast → light_map_clear
→ light_add → sun → godray_seed → godray_march → main`

Of these, the following are **screen-space 2D techniques that do not
generalise**:

- **Shadows** (`shadow.wgsl`): renders walls into a mask, then marches each
  *screen pixel in UV space* toward the sun (16 taps). A 2D top-down occlusion
  trick. The 3D equivalent is shadow mapping (light-space depth render +
  comparison sampler) — different data, different passes, no shared code.
  Note this pass is **already inert** (`frame.rs:374-376`: "Currently unused
  by the on-foot view — the light pipeline supersedes it"); it stays alive
  only because it owns the `wall_mask` the other passes read.

**All four effects are coupled through one shared `wall_mask` texture**
(owned by `ShadowPass`, `shadow.rs:70`, rendered at 2× surface size so
off-screen occluders register). Lights, sun and godray all borrow that view,
and resize must rebuild all four in order or the view dangles
(`mod.rs:171-215`). The mask is a **flat binary occluder map with no height
information to give** — which is precisely why none of these techniques can be
extended to 3D rather than replaced.
- **Lights** (`frame.rs:411/435`): one full render pass *per light*, each a
  fullscreen triangle doing up to 12 raycast taps. In 3D this is replaced by
  forward+/clustered/deferred shading. Note the existing TODO measured this
  at 25.1 ms for 384 large lights and *correctly declined* to cluster it —
  that decision is scoped to 2D and would be reopened by 3D.
- **Godray / sun** (`godray.wgsl`, `sun.wgsl`): screen-space radial
  scattering from a 2D `sun_dir_screen: Vec2`. Has a 3D analogue, but it is
  a rewrite against a depth buffer, not a port.

**The public API is 2D-typed throughout**, which is why these cannot be
quietly swapped underneath: `mod.rs:245` `raycast_and_composite_shadows(sun_dir: Vec2, shadow_length_px: f32)`,
`:344` `queue_sun_pass(sun_dir_screen: Vec2, …)`, `:278` `push_light(…)` in
screen/world 2D coords. Changing these breaks both shipped games.

### The depth buffer is the gate

Verified at this commit: **14 pipelines across 6 files** (`godray` 3,
`init` 1, `lights` 2, `postprocess` 3, `shadow` 3, `sun` 2) every one
`depth_stencil: None`, and **11 render passes** with
`depth_stencil_attachment: None`. No pipeline sets any `cull_mode` either, and
there is no MSAA.

**Resolved 2026-09-21 — and the 14 figure was a red herring.** Only 3 of
those pipelines take `Vertex::desc()`, and only `main_pipeline` needed depth.
The existing TODO R5 entry deferred this work on a 14-pipeline blast radius
that did not survive contact; both it and Phase 0 below now record the real
number. Depth is in, and it is a no-op in 2D by construction.

Worth noting the real per-frame pass count is **11 fixed passes plus one
render pass per light** (up to `MAX_LIGHTS_PER_FRAME = 384`), all in one
encoder and one submit. Ordering today is painter's-algorithm via a
"split index" scheme (`frame.rs:561-677`) that chops the main batch's index
range and interleaves fullscreen composites. **That machinery is what a depth
buffer would partly replace** — which is why Phase 0 has standalone value for
2D draw-order correctness.

## 4. Cost by subsystem — measured

### Renderer (the bulk)

`Vertex` (`batch.rs`) is **84 bytes across 9 attributes**: `pos[2], uv[2],
color[4], pattern[2], material, overlay, ink[4], scale, local[4]`. This is not
a thin vertex. Several attributes encode *2D-specific* assumptions:

- `pattern` is "world metres in pattern space" so procedural materials stay
  locked to the ground under pan/zoom — a concept that assumes a flat plane.
- `local` is "position within the tile, -1..1" for intra-tile blending.
- `scale` is metres-per-pixel, used to fade hatching before it aliases —
  meaningful only under orthographic projection where one such value exists
  per frame.

Under perspective, `scale` is per-fragment, not per-vertex. **The procedural
material system does not survive a move to perspective unchanged.** This is
the least obvious finding in this document and the easiest to underestimate.
`shader.wgsl` carries ~20 procedural pattern functions (`pat_granite`,
`pat_shale`, …), all functions of a 2D pattern-space coordinate.

**Do not change `Vertex`'s size casually.** Two buffer caps are *derived* from
`size_of::<Vertex>()` (`frame.rs:61`, `:610-611`): `MAX_VERTS = 256MB/84`, and
an index cap of `MAX_VERTS * 3`. `frame.rs:37-43` and `:595-601` record two
separate incidents where these were hardcoded and drifted out of sync with the
real stride — once producing a guard that passed and then panicked, once
letting a frame draw from never-uploaded buffer contents. A widened vertex
moves both numbers. This is the strongest single argument for a separate
`Vertex3D` rather than adding `z` to `Vertex`.

**There is a second load-bearing assumption beyond the missing depth buffer:**
`Batch` pre-transforms every primitive on the CPU into camera-relative *pixel*
space (`camera.rs:70-73`, `batch.rs:545` reconstructs world position backwards
from it), then bakes everything into one monolithic vertex buffer per frame.
**There is no per-object transform anywhere in the pipeline to hang a model
matrix on**, and no instance buffer exists. 3D meshes need per-object
transforms, so this is new architecture rather than a modification.

### Simulation — cheaper than the grep suggests

778 `Vec2`/`DVec2` occurrences engine-wide, but they concentrate where it
matters least:

| Module | count | note |
| --- | --- | --- |
| `ui` | 188 | **screen-space — unaffected by 3D** |
| `collision` | 181 | real work |
| `terrain` | 143 | heightfield; partly 2.5D already |
| `renderer` | 93 | real work |
| `fx` | 69 | real work |
| everything else | ~104 | |

So ~188 of the 778 are UI and simply do not care — all of `src/ui/` and
`text.rs` is screen-space `Vec2` with **zero** world→screen projection
anywhere in it. `fx` splits: `particles` and `bubble` are world-space
(`DVec2`) and need work — both emit on a **circle** via `cos/sin`, which
becomes a sphere — while `rings` and `starfield` are screen-space and
unaffected. `floaty_text` is world-anchored but screen-drawn, so only its
anchor widens.

**The `App`/`SimCtx` interface needs no change at all.** `SimCtx` is
`{ world, input, dt }` — no camera, no viewport, no projection — and the words
"camera" and "projection" appear in no `App`/`ClientApp` signature. The game
owns its own camera and calls `world_to_screen_offset` itself. This is what
makes Phase 5 cleanly separable from Phases 0-4.

Rotation is a smaller
surface than expected: only **8 genuine angle-math sites** engine-wide
(`walk.rs:161` `atan2`, `batch.rs` ×5, `shader.wgsl` ×1, plus `radial_menu`
which is screen-space). Matches in `log.rs`/`checkpoint.rs` are *file*
rotation — false positives.

Collision is the single biggest simulation item, and it is *algorithmically*
2D, not merely 2D-typed:

- Cells keyed `(u32, i32, i32)` = (partition, cx, cy) (`grid.rs:33`), with
  **~7 doubly-nested `for cy { for cx { … } }` traversals** that each need a
  third level.
- `query_segment_in` is a **2D grid-DDA** (Amanatides–Woo, `grid.rs:534-575`)
  — `step_x/step_y`, `t_max_x/t_max_y`, and a "step whichever is smaller"
  branch. The 3D extension is known but is a real rewrite of that branch.
- Narrowphase `obb_vs_obb` is **SAT over exactly 4 axes** (`narrow.rs:100`);
  3D SAT needs **15** (3 + 3 + 9 edge crosses). `circle_vs_obb` hand-writes a
  2×2 rotation (`narrow.rs:134-139`). Half-extents are `[f64; 2]`.
- `obb_axes` derives both axes from a **single scalar rotation** — the place
  where `rot: f32` → `Quat` bites hardest.

Genuinely dimension-free and reusable as-is: the ECS itself (`TypeId`-keyed
storage — a `Transform3D` needs no ECS change), `AoiScratch`, `ColliderId`,
and the generation/slot machinery.

**Also absent, so there is no 2D commitment to undo:** gravity (zero
occurrences in `src/`), forces, constraints, restitution, and any contact
solver. `physics.rs` is 47 lines of semi-implicit Euler plus drag. Its linear
half is dimension-agnostic once the types widen; only the scalar `angular`
needs real thought. A 3D game would likely want a physics layer that does not
exist yet in 2D either — **that is new work, not a port, and it is not
costed here.**

Two other structural 2D spots worth naming: `pathfind.rs` keys nodes on
`(i32, i32)` across three collections with a literal 4-neighbour array at two
sites, and `terrain/noise.rs` hashes 2 lattice ints and bilerps 4 corners
(3D means `hash3` and 8-corner trilerp).

### Netcode — structurally trivial, but not free at scale

`POS_BITS = 16`, `ROT_BITS = 12`, `VEL_BITS = 16` (`snapshot.rs:68-73`).
Adding a Z axis costs:

- **+2 bytes/entity/snapshot** for position, **+2** for velocity — the
  quantized state payload goes 76 → 108 bits, a **42% increase**.
- Rotation is the real change: scalar `rot: f32` → quaternion. At 12 bits/axis
  that is 12 → ~36-48 bits, i.e. **+3-4 bytes/entity**.
- The concrete consequence, from the figures recorded in `snapshot.rs:44-55`:
  **~93 items per datagram falls to ~74**, and a 1834-item keyframe goes from
  34 datagrams to **~42-43**.

Two tests pin the current widths and would need updating: `bitpack.rs:507-516`
asserts "two axes at 16 bits is four bytes", and `bitpack.rs:327` enumerates
the widths `snapshot.rs` actually writes.

**Interest management needs no change to `replication.rs` at all** — that file
contains no geometry, only `EntityId` sets. The 2D-ness lives entirely in
`SpatialGrid::query_circle_into`, i.e. in the collision work already costed
above. Likewise `net/interp.rs` interpolates only ticks and an `f32` blend
factor; it is dimension-agnostic and needs zero changes.

Re-measure with `examples/load_sweep.rs`, which exists for exactly this.

### Persistence

Components derive `Serialize`/`Deserialize` and the registry codec is fully
generic over `T` via serde/bincode (`registry.rs:183-198`) — it never inspects
fields. A **new** `Transform3D` component is therefore additive and
registry-named, and existing save files keep working.

**But widening `Transform2D` in place would be worse than it looks.** bincode
is positional and unversioned per field, so `DVec2`→`DVec3` changes the column
bytes with no self-describing marker. There *is* a per-component
`Entry.version`/`Column.version` check (`registry.rs:128-131`,
`persist/snapshot.rs:192-198`) — but it is a **hard error, not a migration**.
The doc comments in three places promise "the load path runs migrations
forward"; **no migration machinery exists in the tree.** A version bump makes
every existing save fail loudly (`SchemaMismatch`) rather than upgrade. Loud
failure beats corruption, but somebody would have to write the migration path
that the comments already claim exists.

This is an independent and fairly strong argument for the parallel-path
recommendation.

## 5. The downstream constraint — the binding one

Two shipped games consume this engine. Measured at this commit:

| | void-claim | mini-miner-2 |
| --- | --- | --- |
| `Batch` primitive calls (`rect`/`line`/`quad`/`circle`) | 271 | 137 |
| `draw_text` | 347 | 38 |
| `push_quad` (low-level) | **0** | **0** |
| `Transform2D` | 490 | 0 |
| `DVec2` | 1284 | 4 |

**408 primitive call sites and zero `push_quad`.** The TODO's "~2,100" is
high — the real figure is ~793 including text — but the conclusion it draws
is correct and important:

> Both games go entirely through the high-level primitives, so replacing
> `Batch`'s internals behind the same methods moves no call site — and
> changing those methods breaks two shipped games.

Since no game touches `push_quad`, `Batch` internals *can* be reworked freely.
The signatures `rect(center: Vec2, …)` etc. cannot change.

Note void-claim's 1284 `DVec2` and 490 `Transform2D` uses: converting the
engine's component types in place is a **four-figure downstream edit** in a
game this spec has no mandate over.

## 6. Work breakdown

**Phase 0 — Depth buffer (prerequisite). ✅ Done 2026-09-21.**

**The estimate here was wrong and the implementation corrected it.** This
said "14 pipelines, 6 files" — the count of every pipeline in the renderer.
Only **3** take `Vertex::desc()` and draw geometry, and only **1** of those
(`main_pipeline`) draws the scene 3D geometry will join; the other 11 are
fullscreen triangles where depth is meaningless. The offscreen pass renders a
2D sub-scene that is then blurred and the wall mask is a flat binary occluder
map, so neither needs depth — and giving them it would have meant two more
depth textures at two sizes for nothing.

Actual change: **1 pipeline, 1 new module (`src/renderer/depth.rs`), 49 lines
across 4 files.** The state is deliberately a no-op in 2D (`Always`, no
writes), so painter's ordering still decides every pixel.

*Note `tests/materials_render.rs` could not have verified this* — it builds
its own pipeline with `depth_stencil: None` and would have passed however the
wiring went. `tests/depth_buffer.rs` renders the same overlapping geometry
down both paths and requires byte-identical framebuffers; it was verified
load-bearing by flipping the compare to `Less`, which fails with the green
quad winning where blue should.

**Phase 1 — Camera. ✅ Done 2026-09-21.** `Camera3D` beside `Camera2D` in
`renderer/camera.rs`, plus `follow_3d` mirroring `follow`, and `DVec3` added
to `math.rs`'s re-exports.

As predicted, the uniform plumbing needed no change: both cameras fill the
same `CameraUniform`, so they share the bind group layout, buffer and upload
path. A test pins that layout-compatibility, since divergence would silently
invalidate the shared `camera_bgl`.

Two things the estimate did not anticipate, both now encoded in tests:

- **`perspective_rh`, never `perspective_rh_gl`.** The `_gl` variant maps to
  a −1..1 depth range, which would put the near plane *behind* the 1.0 the
  main pass clears to — every fragment fails once a 3D pipeline turns on a
  `Less` test, and the screen goes black with nothing to point at. Verified
  load-bearing: swapping in `_gl` fails the test with the near plane at −1.0.
- **The camera is camera-relative, like the 2D path.** `position` is `DVec3`
  and the view matrix places the eye at the origin, subtracting in `f64`
  before the `f32` cast. Verified load-bearing: casting before subtracting
  loses a 1 m offset entirely at 1e9 m (`got 0`). Vertices fed to this matrix
  must be in the same frame — the contract `world_to_screen_offset` already
  implements for 2D.

Degenerate cases guarded because both silently produce a NaN matrix and a
blank frame rather than an error: a zero-height viewport (minimised window)
and a target sitting on the eye (a follow rig settling).

`Camera3D` is `client`-gated like the rest of `renderer`, but `DVec3` is
exported unconditionally — checked against a headless build, so Phase 5's
simulation work is not blocked by the gate.

**Phase 2 — 3D vertex + pipeline. ✅ Done 2026-09-22.** `Vertex3D` (48 bytes
against the 2D `Vertex`'s 84), `Mesh3D`, `GpuMesh3D`, `shader3d.wgsl` and a
separate pipeline, all behind a new `render3d` feature (implies `client`).
`Vertex` was not touched, for the reasons above.

**This is the first point where 3D actually draws**, and the pipeline here is
what finally uses the Phase 0 depth buffer: `Less` with writes on, against
the 2D path's identity state. Both live in the same main pass, which is legal
because depth state is per-pipeline. 3D geometry draws before the 2D batch so
alpha-blended 2D primitives composite over a finished scene.

Design notes worth carrying forward:

- **The 3D shader is separate, not a branch in `shader.wgsl`.** That file's
  fragment stage is ~20 procedural material functions evaluated in pattern
  space with a per-vertex metres-per-pixel AA fade. Under perspective there
  is no single metres-per-pixel per frame, so those inputs stop meaning
  anything — which is §8's open question, deliberately not answered here.
- **Attribute offsets are derived via `offset_of!`**, unlike `Vertex::desc`'s
  nine hand-written literals. `frame.rs` records two incidents from a
  hardcoded stride drifting; a field reorder now cannot silently produce a
  layout that compiles and renders garbage.
- **A frame is 2D or 3D, not both.** Both cameras share one uniform buffer
  and bind group, so `set_camera_3d` replaces the 2D camera wholesale. A 2D
  HUD over a 3D scene needs a second uniform or a separate pass; that
  decision belongs to whoever first wants it.
- **Back-face culling is on for 3D only.** The 2D path's quad winding has
  never had to be thought about, so culling it would silently drop geometry.

`tests/render3d.rs` draws through the real shader, layout and depth state on
a headless device. Three checks, each verified load-bearing by breaking what
it guards: geometry is visible and lit; geometry *behind* the camera is not
drawn (swapping `look_at_rh`→`look_at_lh` draws 1.98% of the frame and fails
all three); and the nearer of two boxes occludes the farther one drawn after
it (`Less`→`Always` shows blue and fails).

**Phase 3 — Mesh path.** `Mesh` (vertex+index buffers, retained on GPU, not
rebuilt per frame like `Batch`) and a draw API. This is genuinely new
architecture: `Batch` is immediate-mode and rebuilt every frame, which is
wrong for static 3D geometry. Plus a loader (glTF) if meshes come from disk.

**Phase 4 — Lighting/shadows.** Shadow mapping, and a shading model to replace
the per-light fullscreen pass. **The largest and least certain phase.** The
existing light/godray/sun/shadow passes remain for the 2D path.

**Phase 5 — Sim (only if 3D gameplay, not just 3D rendering).** `Transform3D`,
`Velocity3D`, 3D collider shapes (sphere/AABB/capsule), a 3D grid key and DDA,
15-axis SAT, quaternion rotation, Z on the wire. Add these **beside** the 2D
types, not in place of them — in-place widening is what triggers the
save-migration problem in §4 and the four-figure downstream edit in §5.
Independently skippable — see §7. If the game also needs gravity, forces or a
contact solver, note that none of those exist in 2D either: that is new
engine work, not a port, and is not costed here.

**Testing.** `tests/shader_compiles.rs` (naga) and `tests/materials_render.rs`
(headless wgpu pixel readback) already exist and extend naturally to new
shaders and a depth buffer. `benches/hot_paths.rs` guards the query budgets.
The seven CI feature axes all still need to pass; a headless build must link
no new GPU code.

## 7. Recommendation

**Add a parallel 3D path; do not convert the engine.**

1. **Conversion breaks two shipped games** for no benefit to either. Neither
   void-claim nor mini-miner-2 wants 3D; the downstream edit alone is
   four-figure.
2. **The effect chain cannot be ported anyway.** Even a full conversion throws
   away shadow/light/godray/sun and rewrites them. There is no version of this
   where that code is reused — so keeping it working for 2D costs nothing
   extra.
3. **The naming already anticipates this.** `Transform2D`/`Destructible2D`
   leave `Transform3D` free. Whoever named them left the door open.
4. **Phases 0-4 deliver 3D *rendering* without touching the simulation at
   all.** A 3D-rendered game on 2D simulation (isometric, 2.5D, billboarded)
   is a real and common target, and it stops before the expensive Phase 5.

Concretely: a `render3d` feature gating `Vertex3D`, `Camera3D`, `Mesh`, and
the 3D pipelines, beside the existing `client` feature. That matches how the
crate already isolates `wgpu` behind `client`, `net`, `persist` and
`ledger-pg` — this codebase's established and well-executed pattern.

**Honest uncertainty.** Phase 4 is where estimates go wrong; "a shading model
and shadow mapping" spans a weekend to a quarter depending on the quality bar,
and this spec deliberately does not put a number on it. Phase 0 is the one I
would schedule first regardless: it is mechanical, independently testable,
already confirmed by a prior audit, and useful for 2D draw-order correctness
even if 3D is never built.

## 8. Open questions for the owner

1. **3D rendering, or 3D gameplay?** This is the fork that decides whether
   Phase 5 exists, and it roughly doubles the scope.
2. **What quality bar for lighting?** Unlit/vertex-lit is a small Phase 4;
   shadow-mapped PBR is a large one.
3. **Do meshes come from disk?** If yes, a glTF loader and asset pipeline are
   in scope and are not costed here.
4. **What happens to procedural materials?** The `pattern`/`local`/`scale`
   system is a genuinely nice piece of work that assumes a flat plane. Port
   it, drop it for 3D, or run 3D unlit initially?
