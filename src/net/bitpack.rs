//! Bit-level writing and reading for the replication wire format.
//!
//! Nothing in the crate could do this. `bincode` is byte-aligned with
//! fixed-width integers, which is the right trade for a save file — it is
//! written once and read once — and the wrong one for state sent to every
//! client thirty times a second. A boolean costs a byte there and a bit
//! here; an entity index costs four bytes there and usually one here.
//!
//! # Quantisation and why positions are sector-local
//!
//! [`Transform2D::pos`] is [`DVec2`], because the world can span
//! light-hours and `f32` cannot. Sending `f64` would cost 16 bytes per
//! entity per tick and waste most of them: a client cannot see across a
//! light-hour, so the high bits are identical for everything on screen.
//!
//! [`crate::sector`] already solves this for the simulation — a position
//! is a [`Sector2D`] plus an offset kept within ±`sector_size / 2` — and
//! replication reuses it rather than inventing an origin scheme. The
//! offset is what goes on the wire, quantised to a fixed step; the sector
//! travels with the entity's spawn and again whenever it changes, which
//! is rare. Precision is then a property of the sector size and the bit
//! count, not of how far from the origin the entity happens to be.
//!
//! [`Transform2D::pos`]: crate::components::Transform2D::pos
//! [`DVec2`]: glam::DVec2
//! [`Sector2D`]: crate::sector::Sector2D
//!
//! # Reading is fallible, writing is not
//!
//! A writer owns its buffer and cannot fail. A reader is parsing bytes
//! that arrived over a network from a peer that may be broken, hostile,
//! or simply a version behind, so every read returns [`Result`]. Running
//! off the end of the buffer is a normal error here, not a panic — the
//! same reasoning behind `framing::read_msg` taking a mandatory cap.

/// Why a read failed.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BitError {
    /// Ran past the end of the buffer. The packet is truncated or the
    /// two ends disagree about the format.
    Truncated,
    /// A varint's continuation bits ran longer than the type allows,
    /// which means the stream is misaligned rather than merely short.
    VarintTooLong,
    /// A quantised value decoded outside the range its parameters allow.
    OutOfRange,
}

impl std::fmt::Display for BitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated      => write!(f, "packet ended mid-value"),
            Self::VarintTooLong  => write!(f, "varint longer than its type permits"),
            Self::OutOfRange     => write!(f, "quantised value outside its declared range"),
        }
    }
}

impl std::error::Error for BitError {}

/// Writes values into a growable bit buffer, most-significant bit first.
///
/// MSB-first so a hexdump of the output reads in the same order as the
/// code that wrote it, which matters more than it sounds when the only
/// tool for diagnosing a wire bug is a packet capture.
#[derive(Default)]
pub struct BitWriter {
    bytes: Vec<u8>,
    /// Bits used in the final byte, 0..8. Zero means the buffer is
    /// byte-aligned and a new byte is needed for the next write.
    bits_in_last: u8,
}

impl BitWriter {
    pub fn new() -> Self { Self::default() }

    /// Reuse an existing allocation. The replication loop builds one
    /// packet per client per tick, so this is the hot path.
    pub fn clear(&mut self) {
        self.bytes.clear();
        self.bits_in_last = 0;
    }

    /// Bits written so far.
    pub fn bit_len(&self) -> usize {
        if self.bits_in_last == 0 {
            self.bytes.len() * 8
        } else {
            (self.bytes.len() - 1) * 8 + self.bits_in_last as usize
        }
    }

    /// Bytes the finished buffer occupies, including any partial byte.
    pub fn byte_len(&self) -> usize { self.bytes.len() }

    /// Write the low `count` bits of `value`, most significant first.
    ///
    /// `count` above 64 is a caller bug and panics; a wire format is
    /// fixed at compile time, so this cannot be driven by remote input.
    pub fn write_bits(&mut self, value: u64, count: u32) {
        assert!(count <= 64, "cannot write {count} bits from a u64");
        for i in (0..count).rev() {
            self.write_bit((value >> i) & 1 == 1);
        }
    }

    /// Write a single bit. A boolean costs exactly this.
    pub fn write_bit(&mut self, set: bool) {
        if self.bits_in_last == 0 {
            self.bytes.push(0);
        }
        if set {
            let last = self.bytes.len() - 1;
            self.bytes[last] |= 1 << (7 - self.bits_in_last);
        }
        self.bits_in_last = (self.bits_in_last + 1) % 8;
    }

    /// Write a `u64` as a 7-bits-per-byte varint.
    ///
    /// Small numbers cost one byte, which is what makes an entity index
    /// cheap: most worlds never reach the 2^21 entities where this stops
    /// beating a fixed `u32`.
    pub fn write_varint(&mut self, mut value: u64) {
        loop {
            let byte = (value & 0x7f) as u8;
            value >>= 7;
            if value == 0 {
                self.write_bits(byte as u64, 8);
                return;
            }
            self.write_bits((byte | 0x80) as u64, 8);
        }
    }

    /// Write a signed value zig-zag encoded, so small negatives are as
    /// cheap as small positives.
    pub fn write_varint_signed(&mut self, value: i64) {
        self.write_varint(((value << 1) ^ (value >> 63)) as u64);
    }

    /// Write a sector-local coordinate quantised to `bits`.
    ///
    /// `half_extent` is half the sector size: the offset is valid over
    /// `[-half_extent, +half_extent]`, matching what `sector::wrap_pos`
    /// guarantees. Values outside are clamped rather than rejected —
    /// a physics step can overshoot fractionally before the next wrap,
    /// and dropping an entity's position for that is worse than a
    /// sub-quantum error on one tick.
    pub fn write_quantised(&mut self, value: f64, half_extent: f64, bits: u32) {
        debug_assert!((1..=32).contains(&bits), "quantisation needs 1..=32 bits, got {bits}");
        debug_assert!(half_extent > 0.0, "half_extent must be positive");
        let levels = ((1u64 << bits) - 1) as f64;
        let t = ((value / half_extent).clamp(-1.0, 1.0) + 1.0) * 0.5;
        self.write_bits((t * levels).round() as u64, bits);
    }

    /// Finish, padding the last byte with zeros.
    pub fn finish(self) -> Vec<u8> { self.bytes }

    /// Borrow the buffer without consuming the writer, for a caller that
    /// encodes into a reused writer each tick.
    pub fn as_bytes(&self) -> &[u8] { &self.bytes }
}

/// Reads values written by a [`BitWriter`].
pub struct BitReader<'a> {
    bytes: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self { Self { bytes, bit_pos: 0 } }

    /// Bits consumed so far.
    pub fn bit_pos(&self) -> usize { self.bit_pos }

    /// Bits left unread, including the final byte's padding.
    pub fn bits_remaining(&self) -> usize {
        self.bytes.len() * 8 - self.bit_pos
    }

    /// Read one bit.
    pub fn read_bit(&mut self) -> Result<bool, BitError> {
        if self.bit_pos >= self.bytes.len() * 8 {
            return Err(BitError::Truncated);
        }
        let byte = self.bytes[self.bit_pos / 8];
        let bit = (byte >> (7 - (self.bit_pos % 8))) & 1 == 1;
        self.bit_pos += 1;
        Ok(bit)
    }

    /// Read `count` bits into the low bits of a `u64`.
    pub fn read_bits(&mut self, count: u32) -> Result<u64, BitError> {
        assert!(count <= 64, "cannot read {count} bits into a u64");
        let mut out = 0u64;
        for _ in 0..count {
            out = (out << 1) | self.read_bit()? as u64;
        }
        Ok(out)
    }

    /// Read a varint written by [`BitWriter::write_varint`].
    pub fn read_varint(&mut self) -> Result<u64, BitError> {
        let mut out = 0u64;
        for shift in (0..64).step_by(7) {
            let byte = self.read_bits(8)? as u8;
            // The 10th byte can only carry one meaningful bit; anything
            // past that means the stream is misaligned, not just long.
            if shift >= 63 && (byte & 0x7f) > 1 {
                return Err(BitError::VarintTooLong);
            }
            out |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 {
                return Ok(out);
            }
        }
        Err(BitError::VarintTooLong)
    }

    /// Read a zig-zag signed varint.
    pub fn read_varint_signed(&mut self) -> Result<i64, BitError> {
        let raw = self.read_varint()?;
        Ok(((raw >> 1) as i64) ^ -((raw & 1) as i64))
    }

    /// Read a value written by [`BitWriter::write_quantised`].
    pub fn read_quantised(&mut self, half_extent: f64, bits: u32) -> Result<f64, BitError> {
        let levels = ((1u64 << bits) - 1) as f64;
        let raw = self.read_bits(bits)? as f64;
        if raw > levels {
            return Err(BitError::OutOfRange);
        }
        Ok(((raw / levels) * 2.0 - 1.0) * half_extent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_round_trip_in_order() {
        let mut w = BitWriter::new();
        w.write_bit(true);
        w.write_bit(false);
        w.write_bits(0b1011, 4);
        assert_eq!(w.bit_len(), 6);

        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        assert!(r.read_bit().unwrap());
        assert!(!r.read_bit().unwrap());
        assert_eq!(r.read_bits(4).unwrap(), 0b1011);
    }

    /// A boolean is one bit, which is the entire reason this module
    /// exists rather than reusing bincode.
    #[test]
    fn eight_booleans_fit_in_one_byte() {
        let mut w = BitWriter::new();
        for i in 0..8 { w.write_bit(i % 2 == 0); }
        assert_eq!(w.byte_len(), 1, "eight bits is one byte, not eight");

        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        for i in 0..8 {
            assert_eq!(r.read_bit().unwrap(), i % 2 == 0, "bit {i}");
        }
    }

    #[test]
    fn varints_are_small_for_small_numbers() {
        for (value, expect_bytes) in [(0u64, 1usize), (127, 1), (128, 2), (16_383, 2), (16_384, 3)] {
            let mut w = BitWriter::new();
            w.write_varint(value);
            assert_eq!(w.byte_len(), expect_bytes, "varint({value})");

            let bytes = w.finish();
            assert_eq!(BitReader::new(&bytes).read_varint().unwrap(), value);
        }
    }

    #[test]
    fn varints_round_trip_at_the_extremes() {
        for value in [0u64, 1, u32::MAX as u64, u64::MAX - 1, u64::MAX] {
            let mut w = BitWriter::new();
            w.write_varint(value);
            let bytes = w.finish();
            assert_eq!(BitReader::new(&bytes).read_varint().unwrap(), value, "{value}");
        }
    }

    /// Zig-zag is what keeps a small negative delta as cheap as a small
    /// positive one — the common case for velocity and position deltas.
    #[test]
    fn signed_varints_are_symmetric() {
        let mut small = BitWriter::new();
        small.write_varint_signed(-1);
        assert_eq!(small.byte_len(), 1, "-1 must cost one byte, not ten");

        for value in [0i64, -1, 1, -64, 63, i64::MIN, i64::MAX] {
            let mut w = BitWriter::new();
            w.write_varint_signed(value);
            let bytes = w.finish();
            assert_eq!(BitReader::new(&bytes).read_varint_signed().unwrap(), value, "{value}");
        }
    }

    /// Quantisation error must stay within one step, which is what makes
    /// a bit count choosable from a precision requirement.
    #[test]
    fn quantised_positions_stay_within_one_step() {
        let half = 500.0;      // a 1000-unit sector
        let bits = 16;
        let step = (2.0 * half) / ((1u64 << bits) - 1) as f64;

        for v in [-500.0, -123.456, -0.5, 0.0, 0.5, 123.456, 499.999, 500.0] {
            let mut w = BitWriter::new();
            w.write_quantised(v, half, bits);
            let bytes = w.finish();
            let got = BitReader::new(&bytes).read_quantised(half, bits).unwrap();
            assert!((got - v).abs() <= step, "v={v} got={got} step={step}");
        }
    }

    /// 16 bits over a 1000-unit sector is ~1.5 cm, and costs two bytes
    /// per axis rather than eight.
    #[test]
    fn sixteen_bits_buys_centimetre_precision_in_a_kilometre_sector() {
        let half = 500.0;
        let step = 1000.0 / ((1u64 << 16) - 1) as f64;
        assert!(step < 0.02, "step was {step} m");

        let mut w = BitWriter::new();
        w.write_quantised(1.0, half, 16);
        w.write_quantised(-1.0, half, 16);
        assert_eq!(w.byte_len(), 4, "two axes at 16 bits is four bytes");
    }

    /// A position that overshoots its sector before the next wrap is
    /// clamped, not dropped: a sub-quantum error beats a missing entity.
    #[test]
    fn out_of_sector_values_clamp_rather_than_wrap() {
        let mut w = BitWriter::new();
        w.write_quantised(900.0, 500.0, 16);
        w.write_quantised(-900.0, 500.0, 16);
        let bytes = w.finish();

        let mut r = BitReader::new(&bytes);
        assert!((r.read_quantised(500.0, 16).unwrap() - 500.0).abs() < 0.02);
        assert!((r.read_quantised(500.0, 16).unwrap() + 500.0).abs() < 0.02);
    }

    /// Reading past the end is an ordinary error. These bytes come from
    /// the network, so a truncated packet must not panic the server.
    #[test]
    fn reading_past_the_end_is_an_error_not_a_panic() {
        let bytes = [0xffu8];
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read_bits(8).unwrap(), 0xff);
        assert_eq!(r.read_bit(), Err(BitError::Truncated));
        assert_eq!(r.read_bits(4), Err(BitError::Truncated));
        assert_eq!(r.read_varint(), Err(BitError::Truncated));
    }

    /// A varint whose continuation bits never terminate is a misaligned
    /// stream, and must be refused rather than read forever.
    #[test]
    fn an_unterminated_varint_is_refused() {
        let bytes = [0x80u8; 12];
        assert_eq!(BitReader::new(&bytes).read_varint(), Err(BitError::VarintTooLong));
    }

    /// Mixed widths must not drift: this is the failure that makes a
    /// wire format decode as garbage three fields later.
    #[test]
    fn unaligned_mixed_widths_stay_in_step() {
        let mut w = BitWriter::new();
        w.write_bit(true);
        w.write_varint(300);
        w.write_bits(0b101, 3);
        w.write_quantised(250.0, 500.0, 16);
        w.write_varint_signed(-42);
        w.write_bit(false);

        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        assert!(r.read_bit().unwrap());
        assert_eq!(r.read_varint().unwrap(), 300);
        assert_eq!(r.read_bits(3).unwrap(), 0b101);
        assert!((r.read_quantised(500.0, 16).unwrap() - 250.0).abs() < 0.02);
        assert_eq!(r.read_varint_signed().unwrap(), -42);
        assert!(!r.read_bit().unwrap());
    }

    /// The writer is reused across clients every tick; clearing must
    /// reset the partial byte too, not just the length.
    #[test]
    fn clear_resets_the_partial_byte() {
        let mut w = BitWriter::new();
        w.write_bits(0b101, 3);
        w.clear();
        assert_eq!(w.bit_len(), 0);
        assert_eq!(w.byte_len(), 0);

        w.write_bit(true);
        assert_eq!(w.bit_len(), 1, "a stale partial byte would offset everything after");
        assert_eq!(w.as_bytes()[0], 0b1000_0000);
    }

    /// An entity index at realistic MMO scale, which is what the varint
    /// is actually for.
    #[test]
    fn an_entity_index_costs_two_bytes_at_mmo_scale() {
        let mut w = BitWriter::new();
        w.write_varint(9_999);
        assert_eq!(w.byte_len(), 2, "a 10k-entity world indexes in two bytes");
        assert_eq!(BitReader::new(w.as_bytes()).read_varint().unwrap(), 9_999);
    }
}
