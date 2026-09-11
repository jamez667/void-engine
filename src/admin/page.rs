//! The HTML page and the JSON document, rendered from one [`Status`].
//!
//! # Escaping is separate from JSON's on purpose
//!
//! [`crate::admin::json`] has its own escaper. These two look similar and
//! must not be merged: they escape different character sets for different
//! grammars. `<` is harmless in JSON and an injection in HTML; `\n` needs
//! an escape in JSON and none in HTML. A single "sanitise" helper is how a
//! value that is safe in one context becomes dangerous in the other.
//!
//! Several strings on this page come from outside the engine — a player
//! id, an asset name like `"item:ore_iron"`, a `reason` from the game's
//! taxonomy, and the free-text reason a Postgres writer went degraded,
//! which is a formatted database error and therefore remote input.

use super::json;
use super::status::{Status, TickSummary};

/// Escape text for HTML body or attribute context.
///
/// All five: the three structural characters plus both quote forms, so
/// the same function is correct inside an attribute as between tags.
pub fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            c => out.push(c),
        }
    }
    out
}

/// The JSON document. Same data as the page, same instant.
pub fn render_json(s: &Status) -> String {
    let zero_sum = json::array(
        &s.zero_sum
            .iter()
            .map(|d| {
                json::object(&[
                    ("asset", json::quote(&d.asset)),
                    // i128 does not fit a JSON number safely, and an
                    // imbalance is read by a human, not arithmetic'd by
                    // the monitoring. A string never loses precision.
                    ("imbalance", json::quote(&d.imbalance.to_string())),
                ])
            })
            .collect::<Vec<_>>(),
    );

    let drift = json::array(
        &s.drift
            .iter()
            .map(|d| {
                json::object(&[
                    ("account", json::quote(&format!("{:?}", d.account))),
                    ("asset", json::quote(&d.asset)),
                    ("cached", json::number(d.cached as f64)),
                    ("actual", json::number(d.actual as f64)),
                ])
            })
            .collect::<Vec<_>>(),
    );

    let writer = json::object(&[
        ("state", json::quote(s.writer.state.label())),
        (
            "reason",
            match s.writer.state.reason() {
                Some(why) => json::quote(why),
                None => "null".to_string(),
            },
        ),
        ("journal_depth", json::number(s.writer.journal_depth as f64)),
    ]);

    let tick = match &s.tick {
        Some(t) => json::object(&[
            ("hz", json::number(t.hz)),
            ("target_hz", json::number(t.target_hz)),
            ("realtime_ratio", json::number(t.realtime_ratio())),
            ("dropped_s", json::number(t.dropped_s)),
            ("mean_tick_ms", json::number(t.mean_tick_ms)),
            ("worst_tick_ms", json::number(t.worst_tick_ms)),
            ("keeping_up", if t.keeping_up() { "true" } else { "false" }.to_string()),
        ]),
        None => "null".to_string(),
    };

    json::object(&[
        ("healthy", if s.healthy() { "true" } else { "false" }.to_string()),
        (
            "problems",
            json::array(&s.problems().iter().map(|p| json::quote(p)).collect::<Vec<_>>()),
        ),
        ("zero_sum", zero_sum),
        ("drift", drift),
        ("reservations", json::number(s.reservations as f64)),
        ("lapsed_reservations", json::number(s.lapsed_reservations as f64)),
        ("acked_tick", json::number(s.acked_tick as f64)),
        ("writer", writer),
        ("tick", tick),
    ])
}

/// The page. Self-contained: no external stylesheet, script or font, so
/// it renders on a box with no internet access — which a locked-down
/// game server generally is.
pub fn render_html(s: &Status) -> String {
    let healthy = s.healthy();
    let banner = if healthy {
        "<div class=\"ok\">Everything checks out.</div>".to_string()
    } else {
        let items: String = s
            .problems()
            .iter()
            .map(|p| format!("<li>{}</li>", escape(p)))
            .collect();
        format!("<div class=\"bad\"><strong>Needs attention</strong><ul>{items}</ul></div>")
    };

    let tick_rows = match &s.tick {
        Some(t) => tick_section(t),
        None => "<p class=\"muted\">The loop is not reporting. Set <code>on_health</code> \
                 on <code>HeadlessConfig</code> to measure it.</p>"
            .to_string(),
    };

    let zero_sum = if s.zero_sum.is_empty() {
        "<p class=\"ok-inline\">Every asset sums to zero.</p>".to_string()
    } else {
        let rows: String = s
            .zero_sum
            .iter()
            .map(|d| {
                format!(
                    "<tr><td>{}</td><td class=\"num bad-text\">{}</td></tr>",
                    escape(&d.asset),
                    escape(&d.imbalance.to_string()),
                )
            })
            .collect();
        format!(
            "<p class=\"bad-text\">Value moved outside the transfer API.</p>\
             <table><tr><th>asset</th><th class=\"num\">imbalance</th></tr>{rows}</table>"
        )
    };

    let drift = if s.drift.is_empty() {
        "<p class=\"ok-inline\">Cached balances match the log.</p>".to_string()
    } else {
        let rows: String = s
            .drift
            .iter()
            .map(|d| {
                format!(
                    "<tr><td>{}</td><td>{}</td><td class=\"num\">{}</td>\
                     <td class=\"num\">{}</td></tr>",
                    escape(&format!("{:?}", d.account)),
                    escape(&d.asset),
                    d.cached,
                    d.actual,
                )
            })
            .collect();
        format!(
            "<table><tr><th>account</th><th>asset</th><th class=\"num\">cached</th>\
             <th class=\"num\">log</th></tr>{rows}</table>"
        )
    };

    // The reservation backlog: the number that predicts a stall hours
    // before it happens. Flagged past a threshold where the scan starts
    // to cost real time — 5,000 resident holds measured 30 µs per spend
    // check, 50,000 measured 335 µs.
    let resv_class = if s.reservations >= 5_000 { "num bad-text" } else { "num" };
    let resv_note = if s.lapsed_reservations > 0 {
        format!(
            "<p class=\"muted\">{} lapsed and awaiting a sweep. \
             Every resident hold is scanned on every spend check, lapsed or not.</p>",
            s.lapsed_reservations,
        )
    } else {
        String::new()
    };

    let writer_row = format!(
        "<tr><td>writer</td><td class=\"{}\">{}{}</td></tr>",
        if s.writer.state.is_alarming() { "bad-text" } else { "" },
        escape(s.writer.state.label()),
        match s.writer.state.reason() {
            Some(why) => format!(" — {}", escape(why)),
            None => String::new(),
        },
    );

    format!(
        r#"<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="5">
<title>void_engine — server status</title>
<style>
:root {{ color-scheme: light dark; }}
body {{ font: 14px/1.5 system-ui, -apple-system, Segoe UI, sans-serif;
       margin: 0 auto; padding: 1.5rem; max-width: 52rem; }}
h1 {{ font-size: 1.25rem; margin: 0 0 1rem; }}
h2 {{ font-size: 1rem; margin: 1.75rem 0 .5rem; }}
.ok, .bad {{ padding: .75rem 1rem; border-radius: 6px; margin-bottom: 1rem; }}
.ok  {{ background: #e6f4ea; color: #10502a; }}
.bad {{ background: #fce8e6; color: #8c1d18; }}
.bad ul {{ margin: .5rem 0 0; padding-left: 1.25rem; }}
.bad-text {{ color: #b3261e; font-weight: 600; }}
.ok-inline {{ color: #1e7a3c; }}
.muted {{ color: #666; font-size: .9em; }}
table {{ border-collapse: collapse; width: 100%; margin: .5rem 0; }}
th, td {{ text-align: left; padding: .35rem .6rem; border-bottom: 1px solid #ddd; }}
th {{ font-weight: 600; font-size: .85em; color: #555; }}
.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
code {{ background: rgba(127,127,127,.15); padding: .1em .3em; border-radius: 3px; }}
footer {{ margin-top: 2rem; color: #666; font-size: .85em; }}
@media (prefers-color-scheme: dark) {{
  .ok  {{ background: #0f2a18; color: #7ee2a8; }}
  .bad {{ background: #2d1210; color: #f2b8b5; }}
  th, td {{ border-bottom-color: #333; }}
  .muted, th, footer {{ color: #999; }}
}}
</style>
</head><body>
<h1>void_engine — server status</h1>
{banner}

<h2>Ledger audits</h2>
{zero_sum}
{drift}

<h2>Durability</h2>
<table>
{writer_row}
<tr><td>journal depth</td><td class="num">{journal}</td></tr>
<tr><td>acked tick</td><td class="num">{acked}</td></tr>
</table>

<h2>Reservations</h2>
<table>
<tr><td>resident holds</td><td class="{resv_class}">{resv}</td></tr>
<tr><td>lapsed</td><td class="num">{lapsed}</td></tr>
</table>
{resv_note}

<h2>Simulation loop</h2>
{tick_rows}

<footer>
Same data as <code>/api/status</code>. Refreshes every 5s. Read-only.
</footer>
</body></html>"#,
        banner = banner,
        zero_sum = zero_sum,
        drift = drift,
        writer_row = writer_row,
        journal = s.writer.journal_depth,
        acked = s.acked_tick,
        resv_class = resv_class,
        resv = s.reservations,
        lapsed = s.lapsed_reservations,
        resv_note = resv_note,
        tick_rows = tick_rows,
    )
}

fn tick_section(t: &TickSummary) -> String {
    let ratio = t.realtime_ratio();
    let ratio_class = if t.keeping_up() { "num" } else { "num bad-text" };
    let dropped_class = if t.dropped_s > 0.0 { "num bad-text" } else { "num" };
    format!(
        "<table>\
         <tr><td>tick rate</td><td class=\"{ratio_class}\">{hz:.1} Hz / {target:.0} Hz</td></tr>\
         <tr><td>real time simulated</td><td class=\"{ratio_class}\">{pct:.1}%</td></tr>\
         <tr><td>simulation dropped</td><td class=\"{dropped_class}\">{dropped:.2} s</td></tr>\
         <tr><td>mean tick</td><td class=\"num\">{mean:.2} ms</td></tr>\
         <tr><td>worst tick</td><td class=\"num\">{worst:.2} ms</td></tr>\
         </table>",
        hz = t.hz,
        target = t.target_hz,
        pct = ratio * 100.0,
        dropped = t.dropped_s,
        mean = t.mean_tick_ms,
        worst = t.worst_tick_ms,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::admin::status::{Drift, Writer};
    use crate::persist::ledger::{Account, Discrepancy};

    #[test]
    fn escapes_every_structural_character() {
        assert_eq!(escape("<b>"), "&lt;b&gt;");
        assert_eq!(escape("a&b"), "a&amp;b");
        assert_eq!(escape(r#"a"b"#), "a&quot;b");
        assert_eq!(escape("a'b"), "a&#39;b");
    }

    /// The realistic injection: an asset name is game-supplied, and a
    /// degraded reason is a formatted database error.
    #[test]
    fn a_hostile_asset_name_cannot_inject_script() {
        let s = Status {
            zero_sum: vec![Discrepancy {
                asset: "<script>alert(1)</script>".into(),
                imbalance: 1,
            }],
            ..Default::default()
        };
        let html = render_html(&s);
        assert!(!html.contains("<script>alert(1)</script>"), "raw script tag reached the page");
        assert!(html.contains("&lt;script&gt;"), "it must appear escaped instead");
    }

    #[test]
    fn a_hostile_writer_reason_cannot_inject_script() {
        let s = Status::default()
            .with_writer(Writer::Failed("<img src=x onerror=alert(1)>".into()), 0);
        let html = render_html(&s);
        assert!(!html.contains("<img src=x"), "raw tag reached the page");
        assert!(html.contains("&lt;img src=x"));
    }

    /// The same string must be safe in JSON too, by JSON's own rules.
    #[test]
    fn a_quote_in_a_reason_does_not_break_the_json() {
        let s = Status::default()
            .with_writer(Writer::Degraded(r#"relation "x" missing"#.into()), 3);
        let doc = render_json(&s);
        assert!(doc.contains(r#""reason":"relation \"x\" missing""#), "got {doc}");
    }

    #[test]
    fn a_healthy_status_renders_both_ways() {
        let s = Status::default();
        let html = render_html(&s);
        assert!(html.contains("Everything checks out"));
        assert!(html.starts_with("<!doctype html>"));

        let doc = render_json(&s);
        assert!(doc.contains(r#""healthy":true"#));
        assert!(doc.contains(r#""problems":[]"#));
        assert!(doc.contains(r#""tick":null"#), "an unmeasured loop is null, not zero");
    }

    #[test]
    fn an_unhealthy_status_lists_its_problems_in_both() {
        let s = Status {
            drift: vec![Drift {
                account: Account::player("alice"),
                asset: "credits".into(),
                cached: 10,
                actual: 7,
            }],
            ..Default::default()
        };
        assert!(render_html(&s).contains("Needs attention"));
        let doc = render_json(&s);
        assert!(doc.contains(r#""healthy":false"#));
        assert!(doc.contains("disagree with the log"));
    }

    /// The page and the endpoint must not disagree, which is the whole
    /// reason both render from one `Status`.
    #[test]
    fn the_page_and_the_endpoint_agree_on_health() {
        for s in [
            Status::default(),
            Status {
                zero_sum: vec![Discrepancy { asset: "credits".into(), imbalance: 9 }],
                ..Default::default()
            },
        ] {
            let html_ok = render_html(&s).contains("Everything checks out");
            let json_ok = render_json(&s).contains(r#""healthy":true"#);
            assert_eq!(html_ok, json_ok, "page and endpoint disagree");
        }
    }

    #[test]
    fn tick_health_reaches_both_renderings() {
        let s = Status::default().with_tick(TickSummary {
            hz: 21.0,
            target_hz: 30.0,
            dropped_s: 1.5,
            mean_tick_ms: 40.0,
            worst_tick_ms: 90.0,
        });
        let html = render_html(&s);
        assert!(html.contains("21.0 Hz / 30 Hz"));
        assert!(html.contains("1.50 s"));

        let doc = render_json(&s);
        assert!(doc.contains(r#""keeping_up":false"#));
        assert!(doc.contains(r#""dropped_s":1.5"#));
    }

    /// An i128 imbalance must not lose precision on the way out.
    #[test]
    fn a_huge_imbalance_survives_as_a_string() {
        let s = Status {
            zero_sum: vec![Discrepancy { asset: "credits".into(), imbalance: i128::MAX }],
            ..Default::default()
        };
        let doc = render_json(&s);
        assert!(doc.contains(&i128::MAX.to_string()), "got {doc}");
    }
}
