//! Just enough JSON to emit a status document.
//!
//! `serde` is already a dependency; `serde_json` is not, and this is not
//! the place to add it. The document this module writes has a fixed shape
//! known at compile time — no arbitrary values, no deserialisation, no
//! `#[derive]` needed — so what is actually required is a correct string
//! escaper and a way to join fields. That is small enough to own.
//!
//! # What must be right
//!
//! Escaping. Several fields are strings the engine did not choose: an
//! asset name (`"item:ore_iron"`, game-supplied), a `reason` and `actor`
//! from the game's own taxonomy, a player id, and the free-text reason a
//! Postgres writer went degraded — which is a formatted database error,
//! i.e. remote input. Any of those containing a quote or a control
//! character would produce a document that parses as something other than
//! intended, and `"` is not exotic in an error message.
//!
//! This escapes per RFC 8259: the two mandatory escapes (`"` and `\`),
//! the five shorthand control escapes, and `\u00XX` for every other
//! character below 0x20. Non-ASCII passes through as UTF-8, which is
//! valid JSON and what every parser expects.

/// Escape a string into a JSON string literal, quotes included.
pub fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            // Everything else below 0x20 has no shorthand and must not be
            // emitted raw: a literal control byte inside a string is
            // invalid JSON, and parsers differ in how loudly they say so.
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// A number, written so it is always valid JSON.
///
/// JSON has no `NaN` or `Infinity`, and emitting either produces a
/// document that fails to parse — which for a health endpoint means the
/// monitoring breaks at exactly the moment something is wrong. A
/// non-finite value becomes `null`, which every parser accepts and which
/// reads honestly as "no value".
pub fn number(v: f64) -> String {
    if v.is_finite() {
        // `{}` on f64 gives the shortest round-tripping form, which for
        // whole numbers is `3` rather than `3.0` — both valid JSON.
        format!("{v}")
    } else {
        "null".to_string()
    }
}

/// Build a JSON object from `(key, already-encoded-value)` pairs.
///
/// Values arrive pre-encoded so a caller can nest objects and arrays
/// without this module needing a value type.
pub fn object(fields: &[(&str, String)]) -> String {
    let inner: Vec<String> =
        fields.iter().map(|(k, v)| format!("{}:{}", quote(k), v)).collect();
    format!("{{{}}}", inner.join(","))
}

/// Build a JSON array from already-encoded elements.
pub fn array(items: &[String]) -> String {
    format!("[{}]", items.join(","))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quotes_and_backslashes_are_escaped() {
        assert_eq!(quote(r#"a"b"#), r#""a\"b""#);
        assert_eq!(quote(r"a\b"), r#""a\\b""#);
    }

    /// The realistic case: a database error containing a quote, which is
    /// what `WriterHealth::Degraded` carries.
    #[test]
    fn a_degraded_reason_containing_quotes_stays_one_string() {
        let why = r#"db error: relation "ledger_entries" does not exist"#;
        let doc = object(&[("degraded", quote(why))]);
        // Exactly two unescaped quotes around the value, and the inner
        // ones escaped — otherwise the document ends early and the rest
        // parses as garbage.
        assert!(doc.contains(r#""degraded":"db error: relation \"ledger_entries\" does not exist""#));
    }

    #[test]
    fn control_characters_are_escaped() {
        assert_eq!(quote("a\nb"), r#""a\nb""#);
        assert_eq!(quote("a\tb"), r#""a\tb""#);
        // Spelled as text rather than embedded as literal control bytes:
        // a raw NUL in the source would make the expectation the
        // *unescaped* form and quietly assert the opposite of the point.
        assert_eq!(quote("a\u{0}b"), "\"a\\u0000b\"");
        assert_eq!(quote("a\u{1f}b"), "\"a\\u001fb\"");
        // The two with shorthand escapes that are easy to forget.
        assert_eq!(quote("a\u{8}b"), r#""a\bb""#);
        assert_eq!(quote("a\u{c}b"), r#""a\fb""#);
    }

    /// Non-ASCII is valid JSON as UTF-8 and must not be mangled into
    /// escapes — a player id or an asset name may legitimately contain it.
    #[test]
    fn non_ascii_passes_through() {
        assert_eq!(quote("naïve"), "\"naïve\"");
        assert_eq!(quote("日本語"), "\"日本語\"");
    }

    /// A health endpoint that emits `NaN` produces a document the
    /// monitoring cannot parse, precisely when it matters.
    #[test]
    fn non_finite_numbers_become_null() {
        assert_eq!(number(f64::NAN), "null");
        assert_eq!(number(f64::INFINITY), "null");
        assert_eq!(number(f64::NEG_INFINITY), "null");
        assert_eq!(number(0.0), "0");
        assert_eq!(number(1.5), "1.5");
    }

    #[test]
    fn objects_and_arrays_compose() {
        let doc = object(&[
            ("ok", "true".to_string()),
            ("items", array(&[quote("a"), number(2.0)])),
            ("nested", object(&[("k", quote("v"))])),
        ]);
        assert_eq!(doc, r#"{"ok":true,"items":["a",2],"nested":{"k":"v"}}"#);
    }

    #[test]
    fn an_empty_object_is_still_valid() {
        assert_eq!(object(&[]), "{}");
        assert_eq!(array(&[]), "[]");
    }
}
