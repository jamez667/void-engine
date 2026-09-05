use glam::{DVec2, Mat4, Vec2};
use bytemuck::{Pod, Zeroable};

use crate::{EntityId, World};
use crate::components::Transform2D;

/// Exp-smoothed camera follow. Move `camera_pos` toward the target
/// entity's `Transform2D.pos` with a critically-damped feel; `smoothing`
/// controls how snappy — higher = tighter follow, lower = more lag.
/// void_sim's default is 8.0.
pub fn follow(world: &World, target: EntityId, camera_pos: &mut DVec2, dt: f32, smoothing: f64) {
    if let Some(t) = world.get::<Transform2D>(target) {
        *camera_pos += (t.pos - *camera_pos) * (1.0 - (-smoothing * dt as f64).exp());
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
}

pub struct Camera2D {
    pub position: Vec2,
    pub zoom: f32,
    pub viewport_size: Vec2,
}

impl Camera2D {
    pub fn new(viewport_size: Vec2) -> Self {
        Self {
            position: Vec2::ZERO,
            zoom: 1.0,
            viewport_size,
        }
    }

    pub fn build_uniform(&self) -> CameraUniform {
        let half_w = self.viewport_size.x * 0.5 / self.zoom;
        let half_h = self.viewport_size.y * 0.5 / self.zoom;
        let left = self.position.x - half_w;
        let right = self.position.x + half_w;
        let bottom = self.position.y - half_h;
        let top = self.position.y + half_h;
        let proj = Mat4::orthographic_rh(left, right, bottom, top, -1.0, 1.0);
        CameraUniform {
            view_proj: proj.to_cols_array_2d(),
        }
    }

    pub fn screen_to_world(&self, screen_pos: Vec2) -> Vec2 {
        let ndc = screen_pos / self.viewport_size * 2.0 - Vec2::ONE;
        let half_w = self.viewport_size.x * 0.5 / self.zoom;
        let half_h = self.viewport_size.y * 0.5 / self.zoom;
        Vec2::new(
            self.position.x + ndc.x * half_w,
            self.position.y - ndc.y * half_h,
        )
    }
}

/// Project a world-space `DVec2` position into the game's screen frame
/// (origin = camera, Y-up, one metre = `scale` pixels). This is the
/// per-entity math every ship/asteroid/particle draw call runs before
/// handing coordinates to `Batch`. Separate from `Camera2D::screen_to_world`
/// (which does the inverse for the full viewport) because the game
/// keeps world positions in `f64` and only casts to `f32` after
/// subtracting the camera — a mid-billion-metre asteroid would lose
/// precision if this were done in `f32` all the way.
#[inline]
pub fn world_to_screen_offset(world_pos: DVec2, camera_pos: DVec2, scale: f32) -> Vec2 {
    ((world_pos - camera_pos) * scale as f64).as_vec2()
}
