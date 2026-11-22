"""Render local TreeSHAP contributions with readable labels and a common euro scale."""
from io import BytesIO
from textwrap import shorten

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

from pricing import FEATURES

SURFACE, INK, MUTED = "#FCFCFF", "#171A35", "#535B76"
POSITIVE, NEGATIVE = "#5147C5", "#A3452B"
ESTIMATE_SURFACE = "#F0F0FC"
UNITS = {"Clock speed": "GHz", "Weight": "kg", "Memory": "GB", "Hard-drive storage": "GB",
         "Solid-state storage": "GB", "Screen size": "in", "Pixel density": "ppi",
         "Horizontal pixels": "px", "Vertical pixels": "px"}


def feature_values(row, detail):
    values = {}
    for field, group in detail.groupby("Feature"):
        key = next(k for k, name in FEATURES.items() if name == field)
        if key in row:
            value = row[key]
        else:
            active = group.loc[group["Input"].eq(1), "Indicator"]
            value = active.iat[0] if len(active) else "Other / unseen"
        if field in ("IPS panel", "Touchscreen"):
            values[field] = "Yes" if value else "No"
        elif field in UNITS:
            values[field] = f"{value:g} {UNITS[field]}"
        else:
            values[field] = str(value)
    return values


def ranked_features(detail):
    totals = detail.groupby("Feature")["SHAP (€)"].sum()
    return totals.reindex(totals.abs().sort_values(ascending=False).index)


def signed_euros(amount):
    if round(amount, 2) == 0:
        return "€0.00"
    sign = "+" if amount > 0 else "−"
    return f"{sign}€{abs(amount):,.2f}"


def draw_step(ax, start, end, row, color, max_head):
    if start == end:
        ax.plot(start, row, marker="|", color=MUTED, markersize=10, zorder=3)
        return
    direction = 1 if end > start else -1
    shoulder = end - direction * min(abs(end - start) * .5, max_head)
    points = [(start, row - .18), (shoulder, row - .18), (shoulder, row - .29),
              (end, row), (shoulder, row + .29), (shoulder, row + .18), (start, row + .18)]
    ax.add_patch(Polygon(points, color=color, linewidth=0, zorder=3))


def waterfall_png(baseline, price, totals, labels, limit=7):
    visible = list(totals.iloc[:limit].items())
    if len(totals) > limit:
        count = len(totals) - limit
        visible.append((f"Other {count} features", float(totals.iloc[limit:].sum())))
    steps = [("Model baseline", baseline, baseline, True)]
    running = baseline
    for label, impact in visible:
        steps.append((label, running, running + impact, False))
        running += impact
    steps.append(("Estimated price", price, price, True))

    fig = Figure(figsize=(9.2, 0.40 * len(steps) + .55), dpi=155, facecolor=SURFACE)
    FigureCanvasAgg(fig)
    left, bars, amounts = fig.subplots(1, 3, gridspec_kw={"width_ratios": [1.85, 4.3, .85]})
    fig.subplots_adjust(left=.025, right=.975, top=.965, bottom=.36 / fig.get_figheight(), wspace=.015)
    for ax in (left, bars, amounts):
        ax.set_facecolor(SURFACE)
        ax.set_ylim(len(steps) - .44, -.54)
        ax.set_yticks([])
        ax.spines[:].set_visible(False)
        ax.axhspan(len(steps) - 1.38, len(steps) - .62, color=ESTIMATE_SURFACE, linewidth=0)
    for ax in (left, amounts):
        ax.set_xlim(0, 1)
        ax.set_xticks([])
    endpoints = [value for _, a, b, _ in steps for value in (a, b)]
    span = max(endpoints) - min(endpoints)
    padding = max(6, span * .05)
    bars.set_xlim(min(endpoints) - padding, max(endpoints) + padding)
    max_head = .025 * (span + 2 * padding)
    bars.xaxis.set_major_locator(MaxNLocator(nbins=4))
    bars.xaxis.set_major_formatter(StrMethodFormatter("€{x:,.0f}"))
    bars.tick_params(axis="x", labelsize=9, colors=MUTED, length=0, pad=7)
    bars.grid(axis="x", color="#E6E8F4", linewidth=.8)
    bars.set_axisbelow(True)
    for i, (name, a, b, endpoint) in enumerate(steps):
        color = INK if endpoint else POSITIVE if b > a else NEGATIVE if b < a else MUTED
        font_size = 12 if i == len(steps) - 1 else 11
        left.text(.025, i-.13 if name in labels else i, name, color=INK, va="center",
                  fontsize=font_size, weight="bold" if endpoint else "medium")
        if name in labels:
            left.text(.025, i+.24, shorten(labels[name], width=27, placeholder="…"),
                      va="center", color=MUTED, fontsize=8.4)
        if i == 0:
            bars.scatter(b, i, s=67, facecolor=SURFACE, edgecolor=INK, linewidth=1.65, zorder=4)
        elif endpoint:
            bars.scatter(b, i, s=74, marker="D", color=POSITIVE, zorder=4)
        else:
            draw_step(bars, a, b, i, color, max_head)
        if i < len(steps)-1:
            bars.plot([b, b], [i, i + 1], color="#757E9D", lw=1, ls=":", zorder=2)
        label = f"€{b:,.2f}" if endpoint else signed_euros(b - a)
        amounts.text(.94, i, label, ha="right", va="center", fontsize=font_size, color=color,
                     weight="bold" if endpoint else "medium")
    image = BytesIO()
    fig.savefig(image, format="png")
    return image.getvalue()
