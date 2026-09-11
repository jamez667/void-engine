# void_engine

A small 2D game engine in Rust: wgpu renderer, homegrown ECS, input handling,
fixed-timestep loop, and an immediate-mode UI layer.

Extracted from [void-claim](https://github.com/jamez667/void-claim), where it
had been a workspace crate. It has no dependencies on that game — only crates.io.
The `terrain` module came the same way out of
[idle-city-sim](https://github.com/jamez667/idle-city-sim).

## Use

```toml
[dependencies]
void_engine = { git = "ssh://git@github.com/jamez667/void-engine.git" }
```

Headless consumers (dedicated servers, simulation) turn off the default
features, which drops both `rodio`/`alsa-sys` and the whole GPU stack:

```toml
void_engine = { git = "ssh://git@github.com/jamez667/void-engine.git", default-features = false }
```

That build contains no `wgpu` at all (`cargo tree -e normal` confirms it), so
it compiles in a headless Linux container with no graphics libraries present.

`winit` is the one exception and stays linked unconditionally: `input` and
`keybinds` use its `KeyCode`/`MouseButton` as plain data, and a server
replaying recorded input or loading a player's keybinds still needs to name
keys. It is pure Rust and opens no OS windowing libraries unless a window is
actually created, so it costs a compiled dependency and nothing at runtime.
What survives is the entire simulation surface: `World`, `collision`,
`pathfind`, `terrain`, `physics`, `time`, `rng`, `sector`, `tilegrid`, and —
with `features = ["net"]` — `net`.

### Client and server

Game logic implements `App`, which is renderer-free and compiles into both
builds. A client additionally implements `ClientApp` on the same type, so the
simulation is written exactly once:

```rust
impl App for Game {
    fn init(&mut self, ctx: &mut SimCtx) { /* spawn the world */ }
    fn fixed_update(&mut self, ctx: &mut SimCtx) { /* ctx.world, ctx.dt */ }
}

// Client build only:
impl ClientApp for Game {
    fn render(&mut self, r: &mut Renderer, w: &World, i: &InputState, alpha: f32) { }
}

void_engine::run(game);                    // window + GPU, 60 Hz
void_engine::run_headless(game, || true);  // no window, 30 Hz
```

`SimCtx::dt` is the loop's own step, so the same `fixed_update` is correct at
both rates.

## What's in it

| Module | |
| --- | --- |
| `app` | Fixed-timestep loop, window + event plumbing (`App`, `run`) |
| `ecs` | `World`, `EntityId` — homegrown, not Bevy |
| `renderer` | wgpu batching renderer, shadows, lights, godrays, postprocess |
| `ui` | Immediate-mode widgets: HUD, modals, radial menu, chat, keybinds panel |
| `fx` | Particles, starfield, rings, bubbles, floaty text |
| `input` | Keyboard/mouse state and rebindable keybinds |
| `math`, `render_math`, `physics`, `collision`, `pathfind` | Simulation primitives |
| `terrain` | Procedural worldgen building blocks: seeded noise/fBm, heightfield ramps and erosion operators, rivers |
| `tilegrid`, `tile_collide`, `walk`, `sector`, `world` | World representation |
| `net` | QUIC endpoints, length-prefixed framing, MTU-aware datagram chunking, snapshot interpolation |
| `audio` | MP3 playback (feature-gated) |
| `text`, `time`, `rng`, `log`, `util` | Support |

## ECS

Components are plain `Clone` structs in a `TypeId`-keyed map inside `World` —
no derive macro.

```rust
let id = world.spawn();            // EntityId = u32 index + u32 generation
world.insert(id, Position(x, y));
world.get::<Position>(id);         // Option<&T>
world.iter::<Position>();          // all entities with T
world.iter2::<Position, Velocity>();
world.despawn(id);
```

`iter`/`iter2`/`iter_mut` are lazy: they borrow the liveness arrays and the
component storage separately, so they need neither an intermediate `Vec` nor
any `unsafe`. Query cost therefore scales with what you actually consume, not
with (entities × systems) — which is what it did when each call collected a
fresh raw-pointer `Vec`. `benches/hot_paths.rs` guards the numbers.

## Renderer

All draws go through `Renderer::batch`, which accumulates vertices and flushes
once per frame. Positions are `[f32; 2]` at the batch site because the GPU
requires it; world positions are `DVec2` and are cast to `Vec2` only after the
camera offset is subtracted, so the values stay f32-safe far from the origin.

## Fixed timestep

`App::fixed_update` runs at a fixed rate; `App::render` runs every frame with an
`alpha` interpolation factor. Both are driven by `void_engine::run()`.

## Build

```
cargo build
cargo test
cargo run --example bubble_preview
```

Replication runs against a real QUIC connection, both ends in one process:

```
cargo run --release --no-default-features --features replication \
    --example replication_server
```
