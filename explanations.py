"""Render local TreeSHAP contributions with readable labels and a common euro scale."""
from io import BytesIO
from textwrap import shorten

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

from pricing import FEATURES

INK, MUTED, TEAL, RUST = "#153735", "#526763", "#087b70", "#ae542e"
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

    fig = Figure(figsize=(9.2, 0.54 * len(steps) + .8), dpi=155, facecolor="#ffffff")
    FigureCanvasAgg(fig)
    left, bars, amounts = fig.subplots(1, 3, gridspec_kw={"width_ratios": [2.1, 3.8, 1.05]})
    fig.subplots_adjust(left=.025, right=.98, top=.965, bottom=.10, wspace=.025)
    for ax in (left, bars, amounts):
        ax.set_ylim(len(steps)-.43, -.55)
        ax.set_yticks([])
        ax.spines[:].set_visible(False)
    for ax in (left, amounts):
        ax.set_xlim(0, 1)
        ax.set_xticks([])
    endpoints = [value for _, a, b, _ in steps for value in (a, b)]
    padding = max(18, (max(endpoints)-min(endpoints))*.1)
    bars.set_xlim(min(endpoints)-padding, max(endpoints)+padding)
    bars.xaxis.set_major_locator(MaxNLocator(nbins=4))
    bars.xaxis.set_major_formatter(StrMethodFormatter("€{x:,.0f}"))
    bars.tick_params(axis="x", labelsize=9, colors=MUTED, length=0, pad=7)
    bars.grid(axis="x", color="#e4ece9", linewidth=.8)
    bars.set_axisbelow(True)
    for i, (name, a, b, endpoint) in enumerate(steps):
        color = INK if endpoint else (TEAL if b >= a else RUST)
        left.text(0, i-.10 if name in labels else i, name, color=INK, va="center",
                  fontsize=11, weight="bold" if endpoint else "medium")
        if name in labels:
            left.text(0, i+.24, shorten(labels[name], width=27, placeholder="…"),
                      va="center", color=MUTED, fontsize=8.4)
        if endpoint:
            bars.scatter(b, i, s=77, color=INK, zorder=4)
        else:
            bars.barh(i, b-a, left=a, height=.46, color=color, zorder=3)
        if i < len(steps)-1:
            bars.plot([b, b], [i+.12, i+.84], color="#869c96", lw=1, ls=":", zorder=2)
        label = f"€{b:,.2f}" if endpoint else f"{'+' if b >= a else '−'}€{abs(b-a):,.2f}"
        amounts.text(1, i, label, ha="right", va="center", fontsize=11, color=color,
                     weight="bold" if endpoint else "medium")
    image = BytesIO()
    fig.savefig(image, format="png", facecolor="white")
    return image.getvalue()
