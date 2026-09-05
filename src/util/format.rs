//! Metric auto-scaling number formatters for UI display. Picks the
//! right SI prefix (or light-year) for the magnitude so a `12 000 000 m`
//! value renders as `12.0 Mm`, `2 000 000 m/s` as `2.0 Mm/s`, etc.

/// Format a distance in metres as a human-readable string with the best
/// SI prefix for its magnitude. Falls back to light-years above ~0.1 ly.
pub fn format_dist(m: f64) -> String {
    const LY: f64 = 9_460_730_472_580_800.0;
    const GM: f64 = 1_000_000_000.0;
    const MM: f64 = 1_000_000.0;
    const KM: f64 = 1_000.0;
    let a = m.abs();
    if a >= LY * 0.1        { format!("{:.2} ly",  m / LY) }
    else if a >= GM * 1_000.0 { format!("{:.2} Pm",  m / (GM * 1_000.0)) }
    else if a >= GM          { format!("{:.1} Gm",  m / GM) }
    else if a >= MM          { format!("{:.1} Mm",  m / MM) }
    else if a >= KM          { format!("{:.1} km",  m / KM) }
    else                     { format!("{:.0} m",   m) }
}

/// Format a speed in metres/second with the best SI prefix for its
/// magnitude. Caps at Mm/s (light-speed prefix isn't useful for the
/// gameplay speeds this is used for).
pub fn format_speed(ms: f64) -> String {
    const MM: f64 = 1_000_000.0;
    const KM: f64 = 1_000.0;
    let a = ms.abs();
    if a >= MM      { format!("{:.1} Mm/s", ms / MM) }
    else if a >= KM { format!("{:.1} km/s", ms / KM) }
    else            { format!("{:.0} m/s",  ms) }
}
