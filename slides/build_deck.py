"""Build the stakeholder deck as an editable PowerPoint file, then export the PDF from it.

Every number is read from a file in reports/ and listed in slides/source_map.csv (and in each slide's speaker notes).

Run from the project folder:  python slides/build_deck.py
It writes slides/deck.pptx, slides/source_map.csv and slides/deck.pdf.
The PPTX needs only python-pptx; the PDF export uses Microsoft PowerPoint (Windows), which must be installed.
"""
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from PIL import Image, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import common, viz  # noqa: E402

REPORTS, SLIDES = common.REPORTS, ROOT / "slides"
ASSETS = SLIDES / "assets"
TOTAL = 12
viz.apply_style()

SOURCE_MAP = []


def src(slide, text, file, column):
    """Return `text` for the slide and record where the number came from."""
    SOURCE_MAP.append({"slide": slide, "text": text, "file": file, "column": column})
    return text


def csv(name):
    return pd.read_csv(REPORTS / name)


def pcts(text):
    return [int(x) for x in re.findall(r"(\d+)%", text)]


# ---------------------------------------------------------------- numbers
months = csv("04_metrics_by_month.csv")
total_flights = int(months["flights"].sum())
overall_rate = float((months["late_or_cancelled_rate"] * months["flights"]).sum() / total_flights)
report = csv("01_data_report.csv").set_index("problem")
findings = csv("02_findings.csv")
f_hour, f_domino, f_airline, f_weather, f_cancel = (findings.loc[i, "number"] for i in range(5))
groups = csv("03_groups.csv").set_index("group_id")
first = csv("07_value_first.csv").iloc[0]
second = csv("07_value_second.csv").iloc[0]
weekly = csv("07_weekly_scores.csv")
weekly["week_start"] = pd.to_datetime(weekly["week_start"])
group_errors = csv("06a_group_errors.csv")
calibration = csv("06a_calibration.csv")

morning_pct, evening_pct = pcts(f_hour)[:2]
late_plane_pct, ok_plane_pct = pcts(f_domino)[:2]
airline_best, best_pct, airline_worst, worst_pct = re.search(r"airlines, (.+?) has (\d+)% of flights late or cancelled and (.+?) has (\d+)%", f_airline).groups()
plane_points, weather_points = re.search(r"adds (\d+) points and bad weather adds (\d+)", f_weather).groups()
dom_adj = csv("02_domino_adjusted.csv").iloc[0]
ten_days_pct = int(re.search(r"(\d+)% of all cancellations", f_cancel).group(1))
thin_late = int(re.search(r"(\d+)% of flights late", groups.loc[2, "typical"]).group(1))
july = months.set_index("month").loc[7]
november = months.set_index("month").loc[11]
late_total = round(first["hits"] / first["recall"])
random_hits = round(late_total * first["random_recall"])
full_weeks = weekly[(weekly["prediction"] == "late_or_cancelled") & (weekly["n"] >= 5000)]
predicted_avg = float((calibration["mean_predicted"] * calibration["flights"]).sum() / calibration["flights"].sum())
recall_by_hour = group_errors[(group_errors["dimension"] == "hour_band") & (group_errors["metric"] == "recall")].sort_values("value")
trigger_line = float(first["recheck_line"])          # set in notebook 07 from the validation weeks


# ---------------------------------------------------------------- charts
def prepared(name):
    """Copy a chart from reports/figures into slides/assets, cutting off its own title (the slide has the title)."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    img = Image.open(REPORTS / "figures" / name).convert("RGB")
    ink = (np.asarray(img.convert("L")) < 225).any(axis=1)
    rows = np.flatnonzero(ink)
    end, blank = 0, 0
    for r in range(rows[0], len(ink)):
        blank = 0 if ink[r] else blank + 1
        if blank >= 8:
            end = r
            break
    img.crop((0, end, img.width, img.height)).save(ASSETS / name)
    return ASSETS / name


def draw_gains():
    d = csv("06a_gains_curve.csv")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(d["share_warned"], d["random_picking"], color=viz.GREY, linewidth=1.6, linestyle=(0, (4, 3)), label="Picking flights at random")
    ax.plot(d["share_warned"], d["hour_rule"], color=viz.ORANGE, linewidth=2.2, label="A simple rule: evening flights first")
    ax.plot(d["share_warned"], d["our_model"], color=viz.BLUE, linewidth=2.8, label="Our warning list")
    at20 = d.loc[(d["share_warned"] - 0.2).abs().idxmin(), "our_model"]
    ax.plot(0.2, at20, marker="o", markersize=10, color=viz.BLUE)
    ax.annotate(f"Warn 2 flights in 10,\nreach {at20:.0%} of the late ones", (0.2, at20), xytext=(0.31, at20 - 0.15), fontsize=12, arrowprops=dict(arrowstyle="-", color=viz.INK_2, lw=0.8))
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0)); ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("Share of flights we warn about (riskiest first)")
    ax.set_ylabel("Share of all late or cancelled flights reached")
    ax.legend(loc="lower right")
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS / "deck_gains.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ASSETS / "deck_gains.png"


def draw_weekly():
    w1 = weekly[(weekly["prediction"] == "late_or_cancelled") & (weekly["n"] >= 5000)]
    w2 = weekly[(weekly["prediction"] == "minutes_late") & (weekly["n"] >= 5000)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7))
    ax = axes[0]
    ax.plot(w1["week_start"], w1["value"], color=viz.BLUE, linewidth=2.4, marker="o", markersize=6)
    ax.axhline(trigger_line, color=viz.ORANGE, linewidth=1.4, linestyle=(0, (4, 3)))
    ax.annotate(f"re-check line {trigger_line:.0%}", (w1["week_start"].iloc[0], trigger_line), xytext=(0, -15), textcoords="offset points", fontsize=10, color=viz.INK_2)
    ax.axhline(first["random_recall"], color=viz.GREY, linewidth=1.2, linestyle=(0, (2, 3)))
    ax.annotate("random picking 20%", (w1["week_start"].iloc[0], first["random_recall"]), xytext=(0, 4), textcoords="offset points", fontsize=10, color=viz.INK_2)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0)); ax.set_ylim(0, 0.55)
    ax.set_title("Warning list: late flights reached each week", fontsize=12)
    ax = axes[1]
    ax.plot(w2["week_start"], w2["value"], color=viz.BLUE, linewidth=2.4, marker="o", markersize=6)
    ax.axhline(0, color=viz.ORANGE, linewidth=1.4, linestyle=(0, (4, 3)))
    ax.annotate("no better than a simple guess", (w2["week_start"].iloc[5], 0), xytext=(0, -16), textcoords="offset points", fontsize=10, color=viz.INK_2)
    ax.set_title("Delay forecast: minutes closer than a simple guess", fontsize=12)
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d")); ax.tick_params(axis="x", labelrotation=40)
    fig.tight_layout()
    fig.savefig(ASSETS / "deck_weekly.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ASSETS / "deck_weekly.png"


# ---------------------------------------------------------------- what the owner checked (from the ticked review boxes)
PLAIN = {
    "A1": "re-ran everything from scratch; the checks passed",
    "A2": "recomputed a headline number from the raw data",
    "A3": "recomputed the value of the warning list",
    "A4": "read how the models are kept from using future information",
    "A5": "confirmed the test months stay untouched until the end",
    "A6": "read the leak found by the independent review and how it was fixed",
    "B1": "went through the data-cleaning decisions", "B2": "went through the findings and their checks",
    "B3": "went through the destination groups", "B4": "recomputed a monthly number from the raw data",
    "B5": "went through each column built for the models", "B6": "went through the model results",
    "B7": "went through the two decision cards and checked their numbers", "B8": "read the log of mistakes found along the way",
}


def ticked_checks():
    text = (ROOT / "second-brain" / "review-checklist.md").read_text(encoding="utf-8")
    ids = re.findall(r"- \[x\] \*\*([AB]\d)\.", text)
    out = [PLAIN[i] for i in ids if i.startswith("A") and i in PLAIN]
    b = [i for i in ids if i.startswith("B")]
    out += ["went through all 8 notebooks and the log of mistakes"] if len(b) == 8 else [PLAIN[i] for i in b if i in PLAIN]
    return out, "A6" in ids


# ---------------------------------------------------------------- PowerPoint helpers
NAVY, NAVY2 = "10243F", "1B3558"
SURFACE, INK, INK2 = "FCFCFB", "0B0B0B", "52514E"
BLUE, ORANGE, AQUA = "2A78D6", "C8501F", "1BAF7A"
CARD, LIGHTBLUE = "EEF2F7", "9EC5F4"
FONT = "Calibri"
FONT_FILES = {False: Path(r"C:\Windows\Fonts\calibri.ttf"), True: Path(r"C:\Windows\Fonts\calibrib.ttf")}
WARNINGS = []


def rgb(hex_):
    return RGBColor.from_string(hex_)


def runs_of(text):
    """Split '... <b>bold</b> ...' into (text, bold) pieces."""
    parts = re.split(r"(<b>.*?</b>)", text)
    return [(p[3:-4], True) if p.startswith("<b>") else (p, False) for p in parts if p]


def estimate_height(paragraphs, width_in):
    """Rough height in inches of wrapped text, measured with the real Calibri widths (to catch overflow early)."""
    if not all(path.exists() for path in FONT_FILES.values()):      # the fit estimate needs the Calibri font files; skip it elsewhere
        return 0.0
    total = 0.0
    for p in paragraphs:
        size = p.get("size", 16)
        words = re.sub(r"</?b>", "", p["text"]).split(" ")
        bold = p.get("bold", False)
        font = ImageFont.truetype(str(FONT_FILES[bold]), 100)
        usable = (width_in - p.get("indent", 0)) * 72
        lines, line = 1, ""
        for w in words:
            trial = (line + " " + w).strip()
            if font.getlength(trial) * size / 100 > usable and line:
                lines, line = lines + 1, w
            else:
                line = trial
        total += lines * size * 1.2 / 72 + p.get("after", 0) / 72
    return total


def add_text(slide, x, y, w, h, paragraphs, name, margin=0.0, anchor=MSO_ANCHOR.TOP, shape=None):
    """Add paragraphs (dicts with text, size, color, bold, after, align, bullet) to a text box, or to `shape`."""
    if shape is None:
        shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    shape.name = name
    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(margin)
    tf.margin_top = tf.margin_bottom = Inches(margin * 0.6)
    for i, p in enumerate(paragraphs):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = p.get("align", PP_ALIGN.LEFT)
        para.space_after = Pt(p.get("after", 0))
        para.line_spacing = p.get("line", 1.0)
        if p.get("bullet"):
            pPr = para._p.get_or_add_pPr()
            pPr.set("marL", str(int(Inches(0.25)))); pPr.set("indent", str(-int(Inches(0.25))))
            bu = pPr.makeelement(qn("a:buChar"), {"char": "•"})
            pPr.append(bu)
            p = {**p, "indent": 0.25}
        for text, bold in runs_of(p["text"]):
            run = para.add_run()
            run.text = text
            run.font.name = FONT
            run.font.size = Pt(p.get("size", 16))
            run.font.bold = bold or p.get("bold", False)
            run.font.color.rgb = rgb(p.get("color", INK))
    need = estimate_height(paragraphs, w - 2 * margin)
    if need > h - margin * 1.2 + 0.02:
        WARNINGS.append(f"{name}: needs about {need:.2f} in but the box is {h:.2f} in")
    return shape


def add_card(slide, x, y, w, h, paragraphs, name, fill=CARD, text_margin=0.22, anchor=MSO_ANCHOR.TOP):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    card.adjustments[0] = 0.05
    card.fill.solid(); card.fill.fore_color.rgb = rgb(fill)
    card.line.fill.background()
    card.shadow.inherit = False
    add_text(slide, x, y, w, h, paragraphs, name, margin=text_margin, anchor=anchor, shape=card)
    return card


def add_dot(slide, x, y, d, color, name):
    dot = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    dot.fill.solid(); dot.fill.fore_color.rgb = rgb(color); dot.line.fill.background(); dot.shadow.inherit = False
    dot.name = name
    return dot


def add_picture_fit(slide, path, x, y, w, h, alt):
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    pic = slide.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))
    pic.name = f"Chart: {alt}"
    pic._element.nvPicPr.cNvPr.set("descr", alt)
    return pic


prs = Presentation()
prs.core_properties.title = "Predicting late and cancelled flights"
prs.core_properties.author = "bssam99"
prs.core_properties.last_modified_by = "bssam99"
prs.core_properties.comments = "Built with Claude Code from the files in reports/; see slides/source_map.csv"
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
MX = 0.6   # side margin


def new_slide(n, title, dark=False):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid(); s.background.fill.fore_color.rgb = rgb(NAVY if dark else SURFACE)
    if title:
        add_text(s, MX, 0.42, 12.1, 1.2, [{"text": title, "size": 34, "bold": True, "color": "FFFFFF" if dark else INK, "line": 0.92}], f"Title {n}", anchor=MSO_ANCHOR.TOP)
    muted = LIGHTBLUE if dark else INK2
    add_text(s, MX, 7.02, 8, 0.3, [{"text": "NYC flights 2023 · predicting late and cancelled departures", "size": 10, "color": muted}], f"Footer {n}")
    add_text(s, 11.2, 7.02, 1.53, 0.3, [{"text": f"{n} / {TOTAL}", "size": 10, "color": muted, "align": PP_ALIGN.RIGHT}], f"Slide number {n}")
    return s


def notes(slide, n, say):
    srcs = [m for m in SOURCE_MAP if m["slide"] == n]
    lines = [re.sub(r"</?b>", "", say), "", "Where the numbers come from:"] + [f"- {m['text']}: {m['file']} ({m['column']})" for m in srcs]
    slide.notes_slide.notes_text_frame.text = "\n".join(lines)


def big_stat(slide, value, label, color, say, say_y=3.85, say_h=2.9):
    add_text(slide, MX, 1.75, 4.6, 1.05, [{"text": value, "size": 62, "bold": True, "color": color}], "Big number")
    add_text(slide, MX, 2.85, 4.5, 0.95, [{"text": label, "size": 16, "color": INK2}], "Big number label")
    if say:
        add_text(slide, MX, say_y, 4.5, say_h, [{"text": say, "size": 17, "line": 1.05}], "Message")


def chart_slide(n, title, value, label, color, say, image, alt):
    s = new_slide(n, title)
    big_stat(s, value, label, color, say)
    add_picture_fit(s, image, 5.45, 1.65, 7.3, 5.2, alt)
    notes(s, n, say)
    return s


# ---------------------------------------------------------------- the 12 slides
# 1 title (dark)
s = new_slide(1, None, dark=True)
add_text(s, 0.9, 1.0, 9, 0.4, [{"text": "Travel agency · flights from New York", "size": 20, "bold": True, "color": LIGHTBLUE}], "Kicker")
add_text(s, 0.9, 1.6, 10.8, 2.4, [{"text": "Can we tell which flights will go wrong, two hours before they leave?", "size": 48, "bold": True, "color": "FFFFFF", "line": 0.95}], "Title 1")
add_text(s, 0.9, 4.1, 8.6, 1.0, [{"text": "What we found in the 2023 flights from JFK, LGA and EWR, what we can predict, and what the agency can do about it.", "size": 20, "color": "D6E4F7"}], "Subtitle")
add_text(s, 0.9, 5.35, 6.5, 1.0, [{"text": src(1, f"{total_flights:,}", "04_metrics_by_month.csv", "flights (sum)"), "size": 54, "bold": True, "color": LIGHTBLUE}], "Big number")
add_text(s, 0.9, 6.3, 6.5, 0.4, [{"text": "flights analysed", "size": 18, "color": "D6E4F7"}], "Big number label")
notes(s, 1, "Open with the question the agency cares about: can we know two hours ahead which flights will go wrong?")

# 2 problem
say2 = "If we knew two hours before which flights are at risk, we could warn the client and arrange a backup."
s = new_slide(2, "Nearly 1 flight in 4 leaves late or is cancelled")
big_stat(s, src(2, f"{overall_rate:.1%}", "04_metrics_by_month.csv", "late_or_cancelled_rate, flights (weighted)"), "of flights left more than 15 minutes late, or never left", BLUE, say2, say_y=3.9)
for i, (head, body) in enumerate([("Late means", "More than 15 minutes after the scheduled time, or the flight never leaves."),
                                  ("Why it matters", "A client misses a connection or a meeting."),
                                  ("Our two questions", "Will this flight be late or cancelled? And how many minutes late will it be?")]):
    add_card(s, 5.6, 1.75 + i * 1.65, 7.1, 1.45, [{"text": head, "size": 18, "bold": True, "after": 4}, {"text": body, "size": 17}], f"Card: {head}", anchor=MSO_ANCHOR.MIDDLE)
notes(s, 2, say2)

# 3 data
say3 = f"All {src(3, f'{total_flights:,}', '04_metrics_by_month.csv', 'flights (sum)')} flights were kept."
s = new_slide(3, "We checked the data before trusting it, and lost no flight")
big_stat(s, src(3, f"{len(report)} checks", "01_data_report.csv", "problem (count of rows)"), "run on the data, with a decision for each problem", BLUE, say3, say_y=3.9)
cancelled = src(3, f"{int(report.loc['cancelled', 'rows']):,}", "01_data_report.csv", "rows (cancelled)")
no_plane = src(3, f"{int(report.loc['missing_tailnum', 'rows']):,}", "01_data_report.csv", "rows (missing_tailnum)")
no_weather = src(3, f"{int(report.loc['weather_gap', 'rows']):,}", "01_data_report.csv", "rows (weather_gap)")
for i, (head, body) in enumerate([(f"{cancelled} cancelled flights", "Kept and marked. They are exactly the flights we want to warn about."),
                                  (f"{no_plane} flights with no plane number, {no_weather} with no weather reading", "Nothing was thrown away: the gaps are marked and handled."),
                                  ("What the data cannot tell us", "Only one year, only flights leaving New York, no passengers or prices, no reasons for delays.")]):
    add_card(s, 5.6, 1.75 + i * 1.65, 7.1, 1.45, [{"text": head, "size": 18, "bold": True, "after": 4}, {"text": body, "size": 17}], f"Card: {head[:30]}", anchor=MSO_ANCHOR.MIDDLE)
notes(s, 3, say3)

# 4 hour
say4 = "Flights scheduled in the morning are rarely late. By the evening, delays from earlier flights have piled up. <b>Clients with tight connections should fly in the morning.</b>"
chart_slide(4, "Delays build up through the day",
            f'{src(4, f"{morning_pct}%", "02_findings.csv", "number (row 1)")} → {src(4, f"{evening_pct}%", "02_findings.csv", "number (row 1)")}',
            "late or cancelled: flights at 6-9 am against 7-10 pm", ORANGE, say4, prepared("02_hour.png"), "Late or cancelled share by scheduled departure hour")

# 5 domino
say5 = (f"When the same plane's earlier flight left more than 15 minutes late, <b>{src(5, f'{late_plane_pct}%', '02_findings.csv', 'number (row 2)')}</b> of its next flights are late or cancelled, "
        f"against <b>{src(5, f'{ok_plane_pct}%', '02_findings.csv', 'number (row 2)')}</b> otherwise. Part of that is a shared bad day: within the same day, airport and part of the day, the overall gap of "
        f"{src(5, f'{dom_adj.raw_gap_points:.0f}', '02_domino_adjusted.csv', 'raw_gap_points')} points shrinks to {src(5, f'{dom_adj.same_day_gap_points:.0f}', '02_domino_adjusted.csv', 'same_day_gap_points')}. The agency can see the earlier flight two hours ahead.")
chart_slide(5, "A late earlier flight is a strong warning sign, stronger than bad weather",
            f'+{src(5, plane_points, "02_findings.csv", "number (row 4)")} vs +{src(5, weather_points, "02_findings.csv", "number (row 4)")}',
            "points of extra risk, each measured with the other absent: a late earlier flight against bad weather", ORANGE, say5, prepared("02_weather.png"), "Risk after a late earlier flight and in bad weather")

# 6 airlines
say6 = "Even when we compare flights from the same airport at the same time of day, the gap remains. <b>For a tight connection, prefer the steadier airlines.</b>"
chart_slide(6, "Airlines differ a lot, and airport and time of day do not explain it",
            f'{src(6, f"{worst_pct}%", "02_findings.csv", "number (row 3)")} vs {src(6, f"{best_pct}%", "02_findings.csv", "number (row 3)")}',
            f"late or cancelled: {airline_worst} against {airline_best}", ORANGE, say6, prepared("02_airlines.png"), "Late or cancelled share by airline")

# 7 destination types
steady, build, thin = groups.loc[0], groups.loc[1], groups.loc[2]
s = new_slide(7, "Destinations fall into three types, each needing a different action")
for i, (color, head, body) in enumerate([
        (BLUE, f"{src(7, str(int(steady['size'])), '03_groups.csv', 'size (steady)')} steady routes", "Short delays all day, but a few more cancellations. Keep a backup for tight connections."),
        (ORANGE, f"{src(7, str(int(build['size'])), '03_groups.csv', 'size (build-up)')} afternoon build-up routes", "Fine in the morning, late after noon. Offer a morning flight."),
        (AQUA, f"{src(7, str(int(thin['size'])), '03_groups.csv', 'size (thin)')} thin routes", f"About one flight a day, {src(7, f'{thin_late}%', '03_groups.csv', 'typical (thin)')} of them late. Offer a flexible ticket.")]):
    y = 1.75 + i * 1.7
    add_card(s, MX, y, 4.75, 1.5, [{"text": head, "size": 18, "bold": True, "after": 4}, {"text": body, "size": 16}], f"Card: {head[:28]}", text_margin=0.22 + 0.0, anchor=MSO_ANCHOR.MIDDLE)
    add_dot(s, MX + 4.75 - 0.42, y + 0.18, 0.22, color, f"Colour key {i + 1}")
add_picture_fit(s, prepared("03_time_of_day.png"), 5.65, 1.65, 7.1, 4.75, "Average delay by time of day for the three destination types")
add_text(s, 5.65, 6.45, 7.1, 0.5, [{"text": "The groups are a simplification: they shift when built from other months, and 'thin routes' is the least stable.", "size": 12, "color": INK2}], "Groups caveat")
notes(s, 7, "Destinations were grouped by their delay pattern. Each type gets a different action.")

# 8 months
say8 = (f"In July <b>{src(8, f'{july.cancellation_rate:.1%}', '04_metrics_by_month.csv', 'cancellation_rate (July)')}</b> of flights were cancelled. Just 10 days hold <b>{src(8, f'{ten_days_pct}%', '02_findings.csv', 'number (row 5)')}</b> "
        "of all cancellations up to September, and the weather at the airports does not fully explain them: the cause is not in our data.")
chart_slide(8, "July was the worst month and autumn the calmest",
            f'{src(8, f"{july.late_or_cancelled_rate:.0%}", "04_metrics_by_month.csv", "late_or_cancelled_rate (July)")} vs {src(8, f"{november.late_or_cancelled_rate:.0%}", "04_metrics_by_month.csv", "late_or_cancelled_rate (November)")}',
            "late or cancelled: July against November. These are the monthly numbers to follow.", ORANGE, say8, prepared("04_monthly_metrics.png"), "Four monthly measures of delays and cancellations")

# 9 prediction 1
say9 = (f"In October to December we would have warned <b>{src(9, f'{int(first.flagged):,}', '07_value_first.csv', 'flagged')}</b> flights (2 in 10) and reached <b>{src(9, f'{int(first.hits):,}', '07_value_first.csv', 'hits')}</b> "
        f"late or cancelled ones, against about <b>{src(9, f'{random_hits:,}', '07_value_first.csv', 'hits, recall, random_recall')}</b> by luck. "
        f"{src(9, f'{first.precision * 100:.0f}', '07_value_first.csv', 'precision')} warnings in 100 were right; {src(9, f'{first.base_rate * 100:.0f}', '07_value_first.csv', 'base_rate')} flights in 100 were late.")
s = new_slide(9, "Prediction 1: warn 2 flights in 10 and reach twice as many late ones as luck")
big_stat(s, f'{src(9, f"{first.recall:.0%}", "07_value_first.csv", "recall")} vs {src(9, f"{first.random_recall:.0%}", "07_value_first.csv", "random_recall")}',
         "of the late or cancelled flights reached, against picking at random", BLUE, say9, say_h=2.0)
for i, (fill, tcolor, head, body) in enumerate([("9EC5F4", INK, "Learn and tune", "Jan-Sep"), (BLUE, "FFFFFF", "Test, kept aside", "Oct-Dec")]):
    add_card(s, MX + i * 2.3, 6.0, 2.2, 0.82, [{"text": head, "size": 14, "bold": True, "color": tcolor, "align": PP_ALIGN.CENTER}, {"text": body, "size": 14, "color": tcolor, "align": PP_ALIGN.CENTER}], f"Time split {i + 1}", fill=fill, text_margin=0.06, anchor=MSO_ANCHOR.MIDDLE)
add_picture_fit(s, draw_gains(), 5.45, 1.65, 7.3, 5.2, "Share of late flights reached by the warning list, the simple rule and random picking")
notes(s, 9, say9)

# 10 prediction 2
say10 = (f"On average the predicted delay is only <b>{src(10, f'{second.gain_minutes:.2f}', '07_value_second.csv', 'gain_minutes')} minutes</b> closer than a simple guess, because a few extreme delays cannot be foreseen. "
         "<b>We use it to rank flights for a connection check, not to promise a number of minutes.</b>")
chart_slide(10, "Prediction 2: how late will a flight be? A modest gain, honestly reported",
            f'{src(10, f"{second.lift:.1f}", "07_value_second.csv", "lift")}×',
            f"more long delays (45+ minutes) among the 5% of flights with the highest predicted delay: {src(10, f'{second.precision_long:.0%}', '07_value_second.csv', 'precision_long')} against {src(10, f'{second.base_rate_long:.0%}', '07_value_second.csv', 'base_rate_long')}",
            BLUE, say10, prepared("06b_by_predicted_tenth.png"), "Real delays by tenth of flights sorted by predicted delay")

# 11 decision cards
s = new_slide(11, "Two decision cards: what to do, for how many flights, and when to re-check")
c1 = [{"text": "Card 1 · warn the client", "size": 19, "bold": True, "after": 4},
      {"text": f"<b>Who:</b> the {src(11, '2 in 10', '07_value_first.csv', 'share_flagged')} riskiest flights, {src(11, f'{int(first.flagged):,}', '07_value_first.csv', 'flagged')} in 3 months.", "size": 15, "after": 3},
      {"text": "<b>Do:</b> warn the client and check the connection. A person confirms first.", "size": 15, "after": 3},
      {"text": f"<b>Re-check</b> if the weekly reach falls below <b>{src(11, f'{trigger_line:.0%}', '07_value_first.csv', 'recheck_line')}</b> two weeks in a row. It stayed between {src(11, f'{full_weeks.value.min():.0%}', '07_weekly_scores.csv', 'value (min)')} and {src(11, f'{full_weeks.value.max():.0%}', '07_weekly_scores.csv', 'value (max)')}.", "size": 15}]
c2 = [{"text": "Card 2 · check the connection", "size": 19, "bold": True, "after": 4},
      {"text": f"<b>Who:</b> the {src(11, '1 in 20', '07_value_second.csv', 'flagged')} flights with the highest predicted delay, {src(11, f'{int(second.flagged):,}', '07_value_second.csv', 'flagged')} in 3 months.", "size": 15, "after": 3},
      {"text": "<b>Do:</b> tell the client to expect a long delay and prepare a backup.", "size": 15, "after": 3},
      {"text": "<b>Re-check</b> or drop it if it is no better than a simple guess two weeks in a row: its gain comes and goes.", "size": 15}]
add_card(s, MX, 1.7, 5.95, 2.2, c1, "Card 1 text")
add_card(s, MX + 6.15, 1.7, 5.95, 2.2, c2, "Card 2 text")
add_picture_fit(s, draw_weekly(), MX, 3.98, 12.1, 2.55, "Weekly reach of the warning list and weekly gain of the delay forecast")
add_text(s, MX, 6.55, 12.1, 0.4, [{"text": "<b>Assumptions to confirm with the agency:</b> it can warn about 2 flights in 10 and check 1 in 20. If not, we change one number and everything updates.", "size": 14, "color": INK2}], "Assumption note")
notes(s, 11, "Card 1 is ready to use as a daily warning list. Card 2 helps order the connection checks; its gain is small and fragile, and the card says so.")

# 12 limits, next steps, AI (dark)
morning, evening = recall_by_hour.iloc[0], recall_by_hour.iloc[-1]
checks, fixes_checked = ticked_checks()
if checks:
    owner = [{"text": "What I checked myself:", "size": 14, "bold": True, "color": "FFFFFF", "after": 3}] + [{"text": c, "size": 14, "color": "FFFFFF", "bullet": True, "after": 2} for c in checks[:7]] + (
        [] if fixes_checked else [{"text": "My review ran while the last fixes were being made; I am re-checking those fixes.", "size": 14, "color": "FFFFFF", "after": 0}])
else:
    owner = [{"text": "<b>What I checked myself:</b> my review of the analysis, the numbers and the code is in progress. This list is filled in from the checks I tick.", "size": 14, "color": "FFFFFF"}]
s = new_slide(12, "What we do not know, what comes next, and how AI helped", dark=True)
H = {"size": 20, "bold": True, "color": LIGHTBLUE, "after": 6}
limits = [{**H, "text": "What we do not know"},
          {"text": f"The test months were calm: the model expected <b>{src(12, f'{predicted_avg:.0%}', '06a_calibration.csv', 'mean_predicted, flights (weighted)')}</b> late flights, <b>{src(12, f'{first.base_rate:.0%}', '07_value_first.csv', 'base_rate')}</b> were. We trust the ranking, not the percentages.", "size": 16, "color": "FFFFFF", "bullet": True, "after": 5},
          {"text": f"It reaches only <b>{src(12, f'{morning.value:.0%}', '06a_group_errors.csv', 'value (hour_band recall, morning)')}</b> of the late morning flights, but <b>{src(12, f'{evening.value:.0%}', '06a_group_errors.csv', 'value (hour_band recall, evening)')}</b> of the evening ones.", "size": 16, "color": "FFFFFF", "bullet": True, "after": 5},
          {"text": "One year of data, no prices, no reasons for delays.", "size": 16, "color": "FFFFFF", "bullet": True}]
nxt = [{**H, "text": "What comes next"}] + [{"text": t, "size": 16, "color": "FFFFFF", "bullet": True, "after": 5} for t in
                                      ["Confirm the two capacities with the agency.", "Add a second year, so seasons can be learned.", "Set a separate warning quota for each part of the day.", "Add the agency's own costs to put a value on each warning."]]
ai = [{**H, "text": "How AI helped"},
      {"text": "Claude Code, an AI assistant, wrote the code, ran the analysis and drafted these slides, following my instructions. It also recorded its own mistakes and corrections.", "size": 14, "color": "FFFFFF", "after": 6}] + owner
for i, (content, w) in enumerate([(limits, 3.6), (nxt, 3.0), (ai, 5.13)]):
    x = MX + [0, 3.8, 7.0][i]
    add_card(s, x, 1.75, w, 4.6, content, f"Panel {i + 1}", fill=NAVY2, text_margin=0.25)
notes(s, 12, "Be upfront about the limits. The last panel says exactly what the AI did and what the owner personally checked.")

# ---------------------------------------------------------------- save, source map, PDF
pptx = SLIDES / "deck.pptx"
prs.save(pptx)
pd.DataFrame(SOURCE_MAP).to_csv(SLIDES / "source_map.csv", index=False)
for w in WARNINGS:
    print("TEXT FIT WARNING:", w)

pdf = SLIDES / "deck.pdf"
ps = (f"$ErrorActionPreference='Stop'; $ppt = New-Object -ComObject PowerPoint.Application; "
      f"$pres = $ppt.Presentations.Open('{pptx}', $true, $false, $false); $pres.SaveAs('{pdf}', 32); $pres.Close(); $ppt.Quit()")
try:
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps], check=True, capture_output=True, timeout=240)
    print(f"wrote {pptx.name} ({pptx.stat().st_size // 1024} KB) and {pdf.name} ({pdf.stat().st_size // 1024} KB), {len(SOURCE_MAP)} sourced numbers")
except (OSError, subprocess.SubprocessError) as error:
    print(f"wrote {pptx.name}; the PDF export needs Microsoft PowerPoint ({error})")
