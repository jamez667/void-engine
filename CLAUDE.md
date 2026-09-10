# void_engine

Graphics engine: wgpu renderer, ECS, input, fixed timestep loop.

## ECS

Homegrown, not Bevy. Components = plain `Clone` structs in `TypeId`-keyed `HashMap<TypeId, Box<dyn Any>>` inside `World`. No `#[derive(Component)]` macro.

```rust
world.spawn() -> EntityId          // EntityId = u32 index + u32 generation
world.insert(id, component)
world.get::<T>(id) -> Option<&T>
world.iter::<T>()                  // all entities with T
world.iter2::<A, B>()              // entities with both A and B
world.despawn(id)
```

`iter`/`iter2`/`iter_mut` are lazy and allocation-free. They used to collect
into a raw-pointer `Vec` to dodge borrow rules; splitting the `alive`/
`generations` borrow from the storage borrow removed the need, along with all
`unsafe` in the ECS. Keep them lazy — the collect cost ~83% of query time.

## Renderer

wgpu-based. All draws via `Renderer::batch` (`Batch` struct). Accumulates verts, flushes once/frame.

Positions `[f32; 2]` at batch site (GPU req). World pos `DVec2`, cast to `Vec2` after subtracting camera so values stay f32-safe.

## Fixed timestep

`App::fixed_update` runs at a fixed rate over `SimCtx` (world + input + dt, no
renderer). `ClientApp::render` runs every frame with an `alpha` interp factor.

Two drivers: `run()` (winit window, 60 Hz, needs `client`) and
`run_headless()` (no window, 30 Hz default). Both share the accumulator,
`can_advance` deferral/refund, and input-edge rules.

Tick rate is per-`Timestep` (`Timestep::with_hz`), not a global — read `dt`
from `SimCtx`, never from the `FIXED_DT` constant, or logic breaks at 30 Hz.

## Features

`client` (default) gates `wgpu` and every module needing it: `renderer`,
`ui`, `fx`, `text`, plus `ClientApp`/`run`. `winit` stays
unconditional — `input`/`keybinds` use `KeyCode` as plain data. Headless
builds link no GPU stack.
