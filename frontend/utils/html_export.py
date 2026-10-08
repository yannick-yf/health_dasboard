"""HTML export for the current WADP weekly review."""

from html import escape

import plotly.io as pio

from utils.wadp import CURRENT_DEFICIT_KCAL


def _fmt(value, unit, decimals=1):
    return f"{value:.{decimals}f} {unit}" if value is not None else "N/A"


def _delta(value, prior, unit, decimals=1):
    return f"{value - prior:+.{decimals}f} {unit} vs prior week" if value is not None and prior is not None else "Prior week unavailable"


def render_report_html(report_data: dict) -> str:
    """Render a standalone report with the same metrics as the Streamlit page."""
    r = report_data
    cards = [
        ("Observed WADP deficit", _fmt(r["avg_deficit"], "kcal/day", 0),
         _delta(r["avg_deficit"], r["prior_avg_deficit"], "kcal/day", 0)),
        ("Weight 7-day average", _fmt(r["weight_ma7"], "kg", 2),
         _delta(r["weight_ma7"], r["prior_weight_ma7"], "kg", 2)),
        ("Waist 7-day average", _fmt(r["waist_ma7"], "cm", 2),
         _delta(r["waist_ma7"], r["prior_waist_ma7"], "cm", 2)),
        ("Waist 14-day average", _fmt(r["waist_ma14"], "cm", 2),
         _delta(r["waist_ma14"], r["prior_waist_ma14"], "cm", 2)),
        ("Training sessions", str(r["sessions"]) if r["sessions"] is not None else "N/A",
         f"Prior week: {r['prior_sessions']}" if r["prior_sessions"] is not None else "Training log unavailable"),
        ("Average sleep", _fmt(r["avg_sleep_h"], "h"), ""),
        ("Average steps", f"{r['avg_steps']:,.0f}" if r["avg_steps"] is not None else "N/A", ""),
    ]
    cards_html = "".join(
        f'<div class="card"><small>{escape(label)}</small><strong>{escape(value)}</strong><span>{escape(delta)}</span></div>'
        for label, value, delta in cards
    )
    notes_html = "".join(f"<li>{escape(note)}</li>" for note in r["signals"])
    trend = pio.to_html(r["fig_trend"], full_html=False, include_plotlyjs="cdn",
                        config={"displayModeBar": False, "responsive": True})
    deficit = pio.to_html(r["fig_deficit"], full_html=False, include_plotlyjs=False,
                          config={"displayModeBar": False, "responsive": True})
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Weekly Health Report — {r['week_start']:%b %d}–{r['week_end']:%b %d, %Y}</title>
<style>body{{font:15px system-ui,sans-serif;background:#0f172a;color:#f0f2f6;max-width:1100px;margin:auto;padding:2rem}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:1rem}}
.card{{background:#1e293b;border-radius:10px;padding:1rem;display:flex;flex-direction:column;gap:.4rem}}
small,span,p{{color:#aeb7c7}}strong{{font-size:1.35rem}}section{{margin:1.5rem 0}}li{{margin:.5rem 0}}</style></head>
<body><h1>Weekly Health Report</h1><h2>{r['week_start']:%b %d}–{r['week_end']:%b %d, %Y}</h2>
<p>Observed WADP = 2,000 + Move − intake; positive means deficit. {r['days_available']}/7 health days,
{r['deficit_days']} days with Move and intake. Current target deficit: {CURRENT_DEFICIT_KCAL} kcal/day.
The target is a relative dial; body and strength trends decide changes.</p>
<section class="cards">{cards_html}</section><section><h2>Notes</h2><ul>{notes_html}</ul></section>
<section>{trend}</section><section>{deficit}</section></body></html>"""
