# void_engine as a Smart Coder plugin — scoping spec

**Status:** proposal, not scheduled. Written 2026-09-22 against `a2bf137`.

Depends on host work specified in the Smart Coder repository as
`docs/specs/29-diagnostic-sources.md`. Figures below were measured or read out
of both trees at that commit, not estimated.

---

## 1. The headline

Smart Coder loads plugins as **subprocesses speaking line-delimited JSON** over
stdio (`sc-plugin-proto`, `PROTOCOL_VERSION = 3`). A plugin declares panels and
commands in a handshake manifest, and renders through four content kinds —
`list`, `text`, `form`, `stack`. That spec is explicit: *"No pixels, no colours,
no layout."*

So the obvious shape — void_engine's viewport embedded in an editor panel — is
not available, and is not a gap to be worked around. It was refused on record
when the Claude Code panel asked for an `Overlay` kind, as *"the widget-tree
road by degrees"*.

**The valuable integration turns out not to need it.**

| Capability | Needs a pixel surface in the editor? | Value |
| --- | --- | --- |
| **Shader diagnostics on save** — `.wgsl` errors inline in the Problems panel | No | **High.** Today a broken shader fails at device creation, in front of the player. |
| **Hot reload** — pipeline rebuilt on save | No | **High.** Iteration loop drops from restart-per-edit to save-per-edit. |
| **Live viewport** | Own window (subprocess, no protocol involvement) | Medium. Same shape as Unity/Godot play mode. |
| **Embedded viewport** | Yes — protocol bump | Low. Costs a `Content` kind argued against the host's own spec. |

**Recommendation: build the first three. Do not pursue the fourth.**
The plugin opens its own window and asks the host only for events,
diagnostics and `EditorOpen`. Rationale in §6.

---

## 2. What is already plugin-shaped (the footholds)

### Subprocess isolation solves a version conflict for free

Smart Coder runs **wgpu 27 / iced 0.14**. void_engine runs **wgpu 22**. In one
process this is fatal — two incompatible wgpu versions in one dependency tree.
As separate processes it is a non-issue: each links its own, and the only shared
type is JSON.

This is worth stating plainly because it inverts the usual intuition. The
subprocess boundary the host chose for *safety* reasons is what makes this
integration **possible at all**.

### naga is already a dependency, and already does the work

`naga = { version = "22", features = ["wgsl-in"] }` is a dev-dependency, and
`tests/shader_compiles.rs` already parses and validates `shader.wgsl` *"the way
wgpu will, so a broken pattern is caught by `cargo test`"*.

The error type carries what a diagnostic needs. Measured directly:

```
error: no definition in scope for identifier: 'oops'
  ┌─ wgsl:1:58
labels: [(Some(57..61), "unknown identifier")]
```

`ParseError::labels()` yields `(Span, String)`; `Span::to_range()` gives byte
offsets. Converting a byte offset to 1-based line/column is a scan over the
source — the only subtlety is counting **characters, not bytes**, for the column,
matching the rule the host's own `offset_of` follows.

`emit_to_string` additionally gives the rendered caret block, which is the right
thing to put in a panel's `text` content for the full error.

### The simulation already runs without a GPU

`run_headless`, `App` and `SimCtx` need no window and no device. A plugin that
only validates shaders never creates a GPU context at all — it links
`default-features = false` and stays a text-processing subprocess. The viewport
is a separate, later concern.

---

## 3. What has no equivalent yet (the real cost)

### Shaders are compiled into the binary

Every shader is loaded with `include_str!`:

| File | Site |
| --- | --- |
| `shader.wgsl` | `init.rs:26`, `postprocess.rs:216`, `shadow.rs:274` |
| `shader3d.wgsl` | `render3d.rs:189` |
| `godray.wgsl`, `lights.wgsl`, `blur.wgsl`, `shadow.wgsl`, `sun.wgsl`, `shadow3d.wgsl` | one site each |

`include_str!` is a **compile-time** macro. The bytes are in the executable;
there is no runtime path that reads a `.wgsl` file from disk, and no pipeline
rebuild after construction.

**This is the actual work in this spec.** Diagnostics do not need it — validating
a file the editor just saved is reading a path the plugin was handed. Hot reload
does, and it is the larger half:

- A runtime shader source: `ShaderSource::{Baked, File(PathBuf)}`, defaulting to
  `Baked` so shipped games are unaffected and lose no `include_str!` guarantee.
- Pipeline rebuild on a valid reload. Pipelines are created in `Renderer::new`
  and its helpers; rebuilding means factoring pipeline creation out of
  construction so it can run again against an existing device.
- **Rejecting a bad reload without tearing down the renderer.** Validate through
  `naga` first, and keep the current pipeline if validation fails. A hot reload
  that can kill the running view is worse than no hot reload, because the
  feedback it exists to provide is precisely what it destroys.

Note the shared files: `shader.wgsl` feeds three pipelines. A reload of it must
rebuild all three or none.

### There is no offscreen renderer

`Renderer::new(window: Arc<Window>)` is surface-bound. `tests/depth_buffer.rs:37`
already stands up an adapter with no window, and `ScreenshotData` already does
full RGBA8 readback with no row padding — so both halves exist, but no
constructor joins them.

**Not needed for this spec.** It is listed because it is the prerequisite for an
embedded viewport, and because knowing its cost is what makes §6's
recommendation a decision rather than an assumption.

---

## 4. Cost by component — measured

### The plugin crate (new, small)

A new binary. Not part of the engine's library surface, and it must not become
one: nothing in `src/` gains a dependency on the plugin protocol.

| Piece | Size | Notes |
| --- | --- | --- |
| stdio loop, handshake, manifest | ~120 lines | The shape is fixed; the three shipped plugins are identical here. |
| `naga` error → `Diagnostic` | ~80 lines | Byte offset → line/column, character-counted. |
| Panel content | ~100 lines | A `list` of errors, a `text` block for the full caret output. |
| Viewport process control | ~150 lines | Spawn, hand it saves, reap on `Shutdown`. |

The manifest, concretely:

```rust
Manifest {
    id: "void-engine",
    name: "void engine",
    protocol_version: PROTOCOL_VERSION,           // 3
    panels: vec![PanelDecl { id: "shaders", title: "Shaders" }],
    commands: vec![
        cmd("void.viewport",  "Open viewport"),
        cmd("void.reload",    "Reload shaders"),
        cmd("void.validate",  "Validate all shaders"),
    ],
    capabilities: vec![Capability::FileRead, Capability::Diagnostics,
                       Capability::EditorOpen],
    subscriptions: vec![Subscription::BufferEvents,
                        Subscription::WorkspaceChanged],
}
```

Three capabilities, where the three shipped plugins declare none. That is the
intended result: this is the first plugin that consumes the editor rather than
using it as a terminal.

`Capability::BufferEdit` is **not** requested. A graphics plugin has no business
writing the user's shaders.

### Runtime shader loading (engine, contained)

| Change | Cost |
| --- | --- |
| `ShaderSource` enum + load path | Small. One enum, one `match` at each of ten sites. |
| Pipeline creation factored out of `Renderer::new` | **Medium.** `init.rs` is the bulk of construction and this is real surgery on it. |
| Validate-before-swap | Small — `naga` is already the pattern in `tests/`. |
| `shader.wgsl`'s three dependents rebuilt together | Small, but easy to get wrong. |

Everything is additive and behind the existing `client` feature. A game that
never sets `ShaderSource::File` keeps `include_str!` and the current behaviour
exactly.

### Offscreen renderer (not in this spec)

Listed for the §6 decision only: a `new_offscreen(width, height)` constructor
rendering to a texture instead of a swapchain, reusing the existing readback.
Medium cost, and it buys nothing until there is somewhere to display the result.

---

## 5. The host constraint — the binding one

Three of the four things this plugin needs are **declared by the host protocol
and not implemented by the host**. Verified against every construction site in
`sc-win` and `sc-craft-ui`:

| Needed | Status in the host |
| --- | --- |
| `BufferEvent::Saved` | Defined; **constructed nowhere.** |
| `PublishDiagnostics` → Problems panel | Received, falls through `_ => None`, **discarded.** |
| `Manifest::subscriptions` honoured | Parsed; **read by nothing.** |
| `WorkspaceChanged` | Defined; **constructed nowhere** (all three shipped plugins handle it regardless). |
| `FileRead`, `EditorOpen` | Live. |

And one that will break the plugin on the host's next release:

> `plugin/mod.rs:240` is `manifest.protocol_version != PROTOCOL_VERSION` —
> strict equality, against documentation promising *"the host supports every
> version it has ever shipped"*.

There is a further wrinkle behind `PublishDiagnostics`: the Problems panel is
backed by a single `Option<CompileReport>` written by the compile button. A
plugin publishing into it would clobber `cargo`'s output and be clobbered in
turn. **That is a host design change, not a wiring change**, and it is the
substance of the companion spec.

**None of this is void_engine's to fix.** Sequencing follows in §7, and the host
work is specified separately as Smart Coder spec 29 so it can be scheduled — and
is worth doing — independently of this repository.

---

## 6. Recommendation

**Own window for the viewport. No embedded surface. No protocol bump.**

Nothing in the plugin protocol forbids a plugin opening a window — it is a
subprocess with the user's full privileges, and the host neither knows nor cares.
So the viewport costs *zero protocol surface*, and the alternative costs a
`Content::Image` kind argued against the host's own stated refusal.

The engineering argument is stronger than the political one. An embedded
viewport means base64-encoded frames over JSON at interactive rates, pushed at
whatever cadence the plugin chooses, rendered into a widget the plugin cannot
size or query. A separate window is a real swapchain at real frame rates with
real input. It is also the shape Unity and Godot use for exactly this
relationship, and for the same reason.

What the editor is good at — showing errors against source lines, jumping to a
file at a line — is what the plugin asks it for. What a game engine is good at
— drawing frames — it keeps.

**The engine gains a runtime shader path it should arguably have anyway.**
`include_str!` for every shader means a shader change costs a recompile *and*
the failure still lands at device creation in front of the player. The reload
path with validate-before-swap fixes both, whether or not an editor ever drives
it. That is the part of this work with value independent of Smart Coder.

---

## 7. Work breakdown

Ordered so each step is independently shippable and nothing waits on the other
repository unnecessarily.

### Host (Smart Coder spec 29) — no void_engine dependency

1. `DiagnosticSource` + keyed diagnostic store; Problems panel merges sources.
2. `PublishDiagnostics` wired into it, `safe_join`-gated, bounded.
3. Subscriptions enforced, then `BufferEvent`/`WorkspaceChanged` emitted.
4. Protocol version range, retained per plugin.

### Engine — no host dependency

5. `ShaderSource::{Baked, File}` and the runtime load path.
6. Pipeline creation factored out of `Renderer::new` so it can run twice.
7. Validate-before-swap; the current pipeline survives a bad reload.
8. `shader.wgsl`'s three dependents rebuild together.

### Plugin — needs 1–4 to be useful, 5–8 to hot reload

9. `sc-plugin-void`: handshake, manifest, stdio loop.
10. `naga` errors → `Diagnostic`, published on save.
11. Shaders panel: `list` of errors clickable to `EditorOpen`, `text` for the
    caret block.
12. Viewport subprocess: spawn on `void.viewport`, reload on save, reaped on
    `Shutdown` — within the host's 500 ms shutdown grace, before its process
    tree is killed.

Steps 9–11 deliver the high-value half and need no engine change at all. Step 12
is what 5–8 exist for.

---

## 8. Open questions for the owner

1. **Does the viewport load a game, or a scene file?** A plugin that runs
   arbitrary game code is a different security proposition from one that renders
   a declarative scene. The host grants no sandbox either way, but the answer
   decides whether this plugin is useful for *a* game or for *this* engine's
   development.

2. **Which repository owns `sc-plugin-void`?** In Smart Coder it is a fourth
   first-party plugin and stays in step with protocol changes automatically. In
   void_engine it is the third-party proof the host's spec 25 exit criterion
   asks for, at the cost of tracking `sc-plugin-proto` by hand. **The second is
   the more honest test**, and this spec assumes it.

3. **Does hot reload extend to meshes and scenes, or stay shader-only?**
   `mesh_store.rs` (`a2bf137`) already addresses meshes by generation-checked
   handle, which is most of what a reload needs. Shader-only is the smaller
   promise and the one costed here.

4. **Is the 2D effect chain in scope?** `godray`, `lights`, `sun`, `shadow` are
   the screen-space passes `docs/3d-spec.md` §4 identifies as needing
   *replacement* rather than porting for 3D. Adding runtime reload to shaders
   that may be deleted is work with a short half-life; scoping reload to
   `shader3d.wgsl` and `shadow3d.wgsl` first may be the better order.
