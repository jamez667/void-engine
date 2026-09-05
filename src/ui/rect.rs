//! Screen-space axis-aligned rect with fill/outline helpers.

use glam::Vec2;
use crate::renderer::batch::Batch;

#[derive(Clone, Copy, Debug)]
pub struct UiRect {
    pub min: Vec2,
    pub max: Vec2,
}

impl UiRect {
    pub fn from_center(center: Vec2, size: Vec2) -> Self {
        let h = size * 0.5;
        Self { min: center - h, max: center + h }
    }

    pub fn from_min_size(min: Vec2, size: Vec2) -> Self {
        Self { min, max: min + size }
    }

    pub fn size(self) -> Vec2 { self.max - self.min }
    pub fn center(self) -> Vec2 { (self.min + self.max) * 0.5 }

    pub fn contains(self, pos: Vec2) -> bool {
        pos.x >= self.min.x && pos.x <= self.max.x
            && pos.y >= self.min.y && pos.y <= self.max.y
    }

    pub fn inset(self, pad: f32) -> Self {
        Self {
            min: self.min + Vec2::splat(pad),
            max: self.max - Vec2::splat(pad),
        }
    }

    pub fn fill(self, batch: &mut Batch, color: [f32; 4]) {
        batch.rect(self.center(), self.size(), color);
    }

    pub fn outline(self, batch: &mut Batch, thickness: f32, color: [f32; 4]) {
        let bl = self.min;
        let br = Vec2::new(self.max.x, self.min.y);
        let tr = self.max;
        let tl = Vec2::new(self.min.x, self.max.y);
        batch.line(bl, br, thickness, color);
        batch.line(br, tr, thickness, color);
        batch.line(tr, tl, thickness, color);
        batch.line(tl, bl, thickness, color);
    }
}
