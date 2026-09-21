use glam::{DVec2, DVec3, Mat4, Vec2, Vec3};
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

/// Perspective camera for the 3D path.
///
/// Sits beside [`Camera2D`] rather than replacing it. The two are genuinely
/// different shapes, not one generalised over a flag:
///
/// - `Camera2D` has **no view matrix at all** — the view is folded into the
///   orthographic bounds, and `Vertex.pos` reaching the shader is already a
///   camera-relative *pixel* offset (see [`world_to_screen_offset`]). The
///   projection's only job is pixels → clip space.
/// - `Camera3D` needs a real `proj * view` over **world-space** vertices,
///   because in 3D the camera has an orientation and vertices cannot be
///   pre-transformed on the CPU into a screen frame that no longer exists.
///
/// Both produce the same [`CameraUniform`], so they share the bind group
/// layout, the buffer and the upload path in `frame.rs` unchanged.
///
/// # Precision
///
/// `position` is [`DVec3`] for the reason `Transform2D.pos` is `DVec2`: a
/// world can span light-hours, and an `f32` eye position would lose sub-metre
/// precision long before that. The view matrix is built **camera-relative** —
/// the eye is placed at the origin and only the *direction* to the target is
/// taken in `f32` — so vertices handed to it must likewise be camera-relative.
/// That is the same contract [`world_to_screen_offset`] implements for 2D, and
/// it is what makes [`crate::world::origin_rebase`] unnecessary for the camera
/// itself.
pub struct Camera3D {
    /// Eye position in world space.
    pub position: DVec3,
    /// Point the camera looks at, in world space.
    pub target: DVec3,
    /// Up vector. `Vec3::Z` matches the engine's existing "+Z is up"
    /// convention, established by `terrain::field::hillshade`, which builds
    /// its surface normal as `(-dz/dx, -dz/dy, 1)`.
    pub up: Vec3,
    /// Vertical field of view, in radians.
    pub fov_y: f32,
    /// Near plane. Must be > 0: `Mat4::perspective_rh` divides by it.
    pub z_near: f32,
    /// Far plane. Must be > `z_near`.
    pub z_far: f32,
    pub viewport_size: Vec2,
}

impl Camera3D {
    /// Default near/far. 0.1 m to 10 km spans what a character-scale game
    /// needs without crushing depth precision near the eye — with a
    /// `[0,1]` reversed-Z-less float depth buffer, precision is worst at
    /// the far plane, and a near plane an order of magnitude smaller would
    /// cost more than the extra range is worth.
    pub const DEFAULT_Z_NEAR: f32 = 0.1;
    pub const DEFAULT_Z_FAR: f32 = 10_000.0;
    /// 60° vertical FOV — a conventional default that reads as neither
    /// telephoto nor fisheye at 16:9.
    pub const DEFAULT_FOV_Y: f32 = std::f32::consts::FRAC_PI_3;

    /// A camera at the origin looking down -Y, with +Z up.
    ///
    /// -Y rather than -Z because +Z is already up here; looking down -Z
    /// would put the camera's forward axis and its up axis on the same
    /// line, which `look_at_rh` cannot resolve.
    pub fn new(viewport_size: Vec2) -> Self {
        Self {
            position: DVec3::ZERO,
            target: DVec3::new(0.0, -1.0, 0.0),
            up: Vec3::Z,
            fov_y: Self::DEFAULT_FOV_Y,
            z_near: Self::DEFAULT_Z_NEAR,
            z_far: Self::DEFAULT_Z_FAR,
            viewport_size,
        }
    }

    /// Aspect ratio, guarding the zero-height case a minimised window
    /// produces. `perspective_rh` would otherwise divide by zero and emit a
    /// matrix full of NaN, which silently blanks the frame rather than
    /// failing.
    pub fn aspect_ratio(&self) -> f32 {
        if self.viewport_size.y <= 0.0 {
            1.0
        } else {
            self.viewport_size.x / self.viewport_size.y
        }
    }

    /// Build `proj * view` for upload.
    ///
    /// **The view is camera-relative**: the eye sits at the origin and the
    /// target becomes the `f64` offset between them, cast to `f32` only
    /// after the subtraction. Vertices fed to this matrix must be in the
    /// same frame — world position minus camera position — which is the
    /// contract 2D already follows via [`world_to_screen_offset`].
    ///
    /// `perspective_rh` (not `_gl`) because it produces the `[0, 1]` depth
    /// range WebGPU expects, matching the `Clear(1.0)` far-plane the main
    /// pass uses. `orthographic_rh` was chosen for `Camera2D` for the same
    /// reason.
    pub fn build_uniform(&self) -> CameraUniform {
        let forward = (self.target - self.position).as_vec3();
        // A zero or non-finite forward vector has no orientation to
        // describe; `look_at_rh` would return NaN. Fall back to the
        // default gaze rather than poison the frame.
        let forward = if forward.length_squared() > 0.0 && forward.is_finite() {
            forward
        } else {
            Vec3::new(0.0, -1.0, 0.0)
        };
        let view = Mat4::look_at_rh(Vec3::ZERO, forward, self.up);
        let proj = Mat4::perspective_rh(
            self.fov_y,
            self.aspect_ratio(),
            self.z_near.max(f32::MIN_POSITIVE),
            self.z_far.max(self.z_near + f32::MIN_POSITIVE),
        );
        CameraUniform {
            view_proj: (proj * view).to_cols_array_2d(),
        }
    }

    /// Offset a world-space position into this camera's frame.
    ///
    /// The 3D counterpart to [`world_to_screen_offset`], and the same
    /// precision rule: subtract in `f64`, cast after. Unlike the 2D version
    /// there is no `scale` — perspective division does what pixels-per-metre
    /// did, so the result is in metres rather than pixels.
    #[inline]
    pub fn world_to_camera_offset(&self, world_pos: DVec3) -> Vec3 {
        (world_pos - self.position).as_vec3()
    }
}

/// Exp-smoothed follow for a [`Camera3D`], mirroring [`follow`].
///
/// Moves the eye toward `target_pos` plus `offset` with the same
/// critically-damped feel, and points it at the target. `offset` is the
/// camera's position relative to what it follows — a third-person rig is
/// something like `DVec3::new(0.0, -8.0, 5.0)`: behind on -Y and above on
/// +Z.
pub fn follow_3d(
    target_pos: DVec3,
    offset: DVec3,
    camera: &mut Camera3D,
    dt: f32,
    smoothing: f64,
) {
    let desired = target_pos + offset;
    let t = 1.0 - (-smoothing * dt as f64).exp();
    camera.position += (desired - camera.position) * t;
    // The look target tracks without smoothing: smoothing it too makes the
    // camera lag its own aim and swim during fast turns.
    camera.target = target_pos;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn cam() -> Camera3D {
        Camera3D::new(Vec2::new(1920.0, 1080.0))
    }

    /// The clip-space depth range has to match what the depth buffer is
    /// cleared to, or everything is culled on the first frame and the
    /// screen goes black with nothing to point at.
    ///
    /// `frame.rs` clears depth to 1.0 as the far plane, so the projection
    /// must map far → 1.0 and near → 0.0. That is `perspective_rh`;
    /// `perspective_rh_gl` maps to -1..1 and would put the near plane at
    /// -1, behind the clear value, so every fragment would fail a `Less`
    /// test once 3D turns one on.
    #[test]
    fn the_projection_matches_the_depth_buffers_0_to_1_range() {
        let c = cam();
        let m = Mat4::from_cols_array_2d(&c.build_uniform().view_proj);

        // Points on the view axis at the near and far planes. The camera
        // is at the origin looking down -Y by default.
        let near = m.project_point3(Vec3::new(0.0, -c.z_near, 0.0));
        let far = m.project_point3(Vec3::new(0.0, -c.z_far, 0.0));

        assert!(
            (near.z - 0.0).abs() < 1e-3,
            "near plane should map to depth 0, got {}",
            near.z,
        );
        assert!(
            (far.z - 1.0).abs() < 1e-3,
            "far plane should map to depth 1 (what frame.rs clears to), got {}",
            far.z,
        );
    }

    /// The whole reason `position` is `DVec3`: subtracting in `f64` and
    /// casting after keeps sub-metre detail a billion metres out, which an
    /// `f32` eye position cannot represent at all.
    #[test]
    fn camera_relative_offset_keeps_precision_far_from_the_origin() {
        let mut c = cam();
        c.position = DVec3::new(1.0e9, 0.0, 0.0);

        // One metre beyond the camera, a billion metres from the origin.
        let offset = c.world_to_camera_offset(DVec3::new(1.0e9 + 1.0, 0.0, 0.0));
        assert!(
            (offset.x - 1.0).abs() < 1e-4,
            "a 1 m offset at 1e9 m should survive the f64 subtraction, got {}",
            offset.x,
        );

        // The same arithmetic done in f32 throughout loses the metre
        // entirely — this is the failure the DVec3 exists to prevent.
        let naive = (1.0e9f32 + 1.0) - 1.0e9f32;
        assert!(
            naive < 0.5,
            "precondition: f32 should lose the metre here (got {naive}), \
             otherwise this test proves nothing about why DVec3 is used",
        );
    }

    /// A minimised window gives a zero-height viewport. Left alone that is
    /// a divide-by-zero into a NaN matrix, which blanks the frame silently
    /// rather than failing loudly.
    #[test]
    fn a_zero_height_viewport_does_not_produce_a_nan_matrix() {
        let mut c = cam();
        c.viewport_size = Vec2::new(1920.0, 0.0);
        let m = Mat4::from_cols_array_2d(&c.build_uniform().view_proj);
        assert!(m.is_finite(), "zero-height viewport produced {m:?}");
    }

    /// `look_at_rh` cannot resolve a zero-length forward vector, and a
    /// camera whose target sits exactly on its eye is an ordinary thing to
    /// hit for one frame while a follow rig settles.
    #[test]
    fn a_target_on_top_of_the_eye_does_not_produce_a_nan_matrix() {
        let mut c = cam();
        c.target = c.position;
        let m = Mat4::from_cols_array_2d(&c.build_uniform().view_proj);
        assert!(m.is_finite(), "degenerate gaze produced {m:?}");
    }

    /// Both cameras fill the same uniform, which is what lets them share
    /// the bind group layout, the buffer and the upload in `frame.rs`.
    /// If these ever diverge in size, the shared `camera_bgl` stops being
    /// valid for one of them.
    #[test]
    fn both_cameras_produce_the_same_uniform_shape() {
        let two = Camera2D::new(Vec2::new(1920.0, 1080.0)).build_uniform();
        let three = cam().build_uniform();
        assert_eq!(
            std::mem::size_of_val(&two),
            std::mem::size_of_val(&three),
            "the two cameras must stay layout-compatible: they share one \
             bind group layout and one uniform buffer",
        );
    }

    /// The follow rig should approach its mark and end up looking at what
    /// it follows, not at where it used to be.
    #[test]
    fn follow_3d_approaches_the_offset_and_aims_at_the_target() {
        let mut c = cam();
        c.position = DVec3::ZERO;
        let target = DVec3::new(100.0, 100.0, 0.0);
        let offset = DVec3::new(0.0, -8.0, 5.0);

        for _ in 0..240 {
            follow_3d(target, offset, &mut c, 1.0 / 60.0, 8.0);
        }

        let want = target + offset;
        assert!(
            (c.position - want).length() < 0.1,
            "follow should converge on target+offset; at {:?}, wanted {:?}",
            c.position,
            want,
        );
        assert_eq!(c.target, target, "the camera should aim at what it follows");
    }
}
