#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct EntityId {
    pub index: u32,
    pub generation: u32,
}
