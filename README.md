# void_engine

A small 2D game engine in Rust: wgpu renderer, homegrown ECS, input handling,
fixed-timestep loop, and an immediate-mode UI layer.

Extracted from [void-claim](https://github.com/jamez667/void-claim), where it
had been a workspace crate. It has no dependencies on that game — only crates.io.

## Use

```toml
[dependencies]
void_engine = { git = "ssh://git@github.com/jamez667/void-engine.git" }
```

Headless consumers (dedicated servers, simulation) should turn off the default
`audio` feature, which pulls in `rodio` and therefore `alsa-sys`:

```toml
void_engine = { git = "ssh://git@github.com/jamez667/void-engine.git", default-features = false }
```

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
| `tilegrid`, `tile_collide`, `walk`, `sector`, `world` | World representation |
| `net` | Snapshot interpolation |
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

`iter`/`iter2` use a raw-pointer collect to sidestep borrow rules. That is
intentional, not a bug.

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
