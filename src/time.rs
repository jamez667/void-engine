pub const FIXED_DT: f32 = 1.0 / 60.0;

pub struct Timestep {
    accumulator: f32,
}

impl Timestep {
    pub fn new() -> Self {
        Self { accumulator: 0.0 }
    }

    pub fn advance(&mut self, frame_dt: f32) -> (u32, f32) {
        self.accumulator += frame_dt.min(0.25);
        let steps = (self.accumulator / FIXED_DT) as u32;
        self.accumulator -= steps as f32 * FIXED_DT;
        let alpha = self.accumulator / FIXED_DT;
        (steps, alpha)
    }
}

impl Default for Timestep {
    fn default() -> Self {
        Self::new()
    }
}
