// `DVec3` joins `DVec2` for the 3D path: world positions are `f64` here for
// the same reason they are in 2D — a world can span light-hours, and an
// `f32` position loses sub-metre precision long before that. See
// `renderer::camera::Camera3D` for the camera-relative cast rule.
pub use glam::{Vec2, Vec3, Vec4, Mat4, Quat, DVec2, DVec3};
