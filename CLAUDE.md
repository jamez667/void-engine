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

`iter`/`iter2` use raw-pointer collect to dodge borrow rules — intentional, not bug.

## Renderer

wgpu-based. All draws via `Renderer::batch` (`Batch` struct). Accumulates verts, flushes once/frame.

Positions `[f32; 2]` at batch site (GPU req). World pos `DVec2`, cast to `Vec2` after subtracting camera so values stay f32-safe.

## Fixed timestep

`App::fixed_update` runs fixed rate. `App::render` every frame with `alpha` interp factor. Both driven by `void_engine::run()`.
