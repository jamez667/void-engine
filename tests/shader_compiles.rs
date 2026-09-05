//! The shader is data, not code: a mistake in it fails at device creation, in
//! front of the player, rather than at build time. This parses and validates it
//! the way wgpu will, so a broken pattern is caught by `cargo test`.

#[test]
fn main_shader_is_valid_wgsl() {
    let src = include_str!("../src/renderer/shader.wgsl");
    let module = naga::front::wgsl::parse_str(src).expect("shader must parse as WGSL");
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    );
    validator
        .validate(&module)
        .expect("shader must pass validation");
}
