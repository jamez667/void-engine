//! Push/pop modal stack.
//!
//! Replaces the ad-hoc `if modal_a { close(a) } else if modal_b { close(b) } …`
//! chain that Esc handlers grow into. Every open registers on the stack; Esc
//! pops the top.
//!
//! Generic over the game's `ModalId` — an enum in the game crate.

use std::fmt::Debug;

/// LIFO stack of currently-open modals. `top()` is the front-most; `pop()`
/// removes it. Uniqueness is enforced — pushing an id that's already on the
/// stack moves it to the top rather than duplicating.
#[derive(Clone, Debug, Default)]
pub struct ModalStack<M: Copy + Eq + Debug> {
    modals: Vec<M>,
}

impl<M: Copy + Eq + Debug> ModalStack<M> {
    pub fn new() -> Self { Self { modals: Vec::new() } }

    /// Push `id` onto the top. If already present, remove the old position
    /// first so `top()` always reflects the most recently opened.
    pub fn push(&mut self, id: M) {
        self.modals.retain(|m| *m != id);
        self.modals.push(id);
    }

    /// Pop and return the top id, if any.
    pub fn pop(&mut self) -> Option<M> { self.modals.pop() }

    /// Front-most (last-pushed) id.
    pub fn top(&self) -> Option<M> { self.modals.last().copied() }

    pub fn contains(&self, id: M) -> bool { self.modals.contains(&id) }

    /// Remove `id` from anywhere in the stack. Silently no-op if absent.
    /// Use for close paths that aren't guaranteed to be the top (e.g. a
    /// click on the modal's own X, or a state change that dismisses it).
    pub fn remove(&mut self, id: M) {
        self.modals.retain(|m| *m != id);
    }

    pub fn is_empty(&self) -> bool { self.modals.is_empty() }

    pub fn len(&self) -> usize { self.modals.len() }

    /// Wipe every entry. For teardown / full state reset paths.
    pub fn clear(&mut self) { self.modals.clear(); }

    /// Bottom-to-top slice of currently open modals. Useful for rendering
    /// them in z-order or for debug dumps.
    pub fn as_slice(&self) -> &[M] { &self.modals }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum M { A, B, C }

    #[test]
    fn push_pop_lifo() {
        let mut s = ModalStack::<M>::new();
        s.push(M::A);
        s.push(M::B);
        assert_eq!(s.top(), Some(M::B));
        assert_eq!(s.pop(), Some(M::B));
        assert_eq!(s.top(), Some(M::A));
        assert_eq!(s.pop(), Some(M::A));
        assert_eq!(s.pop(), None);
        assert!(s.is_empty());
    }

    #[test]
    fn push_dedupes_moves_to_top() {
        let mut s = ModalStack::<M>::new();
        s.push(M::A);
        s.push(M::B);
        s.push(M::A); // A moves to top; not duplicated
        assert_eq!(s.len(), 2);
        assert_eq!(s.top(), Some(M::A));
        s.pop();
        assert_eq!(s.top(), Some(M::B));
    }

    #[test]
    fn remove_from_middle() {
        let mut s = ModalStack::<M>::new();
        s.push(M::A);
        s.push(M::B);
        s.push(M::C);
        s.remove(M::B);
        assert_eq!(s.len(), 2);
        assert_eq!(s.top(), Some(M::C));
        assert!(!s.contains(M::B));
    }
}
