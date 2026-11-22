"""Streamlit entry point for the laptop catalog price estimator."""
import math

import pandas as pd
import streamlit as st

from buyer_report import listing_inputs, show_report
from explanations import POSITIVE, feature_values, ranked_features, signed_euros, waterfall_png
from pricing import explain, load_bundle

cached_bundle = st.cache_resource(load_bundle)


def configure(catalog):
    choices = catalog["choices"]

    def choose(label, field, default):
        options = choices[field]
        return st.selectbox(label, options, index=options.index(default))

    with st.sidebar:
        st.header("Laptop specs", anchor=False)
        with st.expander("Performance & storage", icon=":material/memory:", expanded=True):
            cpus, gpus = sorted(catalog["cpus"]), sorted(catalog["gpus"])
            cpu = st.selectbox("Processor", cpus, index=cpus.index("Intel Core i5 7200U 2.5GHz"))
            gpu = st.selectbox("Graphics", gpus, index=gpus.index("Intel HD Graphics 620"))
            ram = choose("Memory (GB)", "Ram", 8)
            ssd = choose("SSD or flash storage (GB)", "SSD_GB", 256)
            hdd = choose("HDD or hybrid storage (GB)", "HDD_GB", 0)
        with st.expander("Device", icon=":material/laptop_windows:"):
            brand = choose("Brand", "Company", "Dell")
            chassis = choose("Form factor", "TypeName", "Notebook")
            os_name = choose("Operating system", "OpSys", "Windows 10")
            lo, hi = catalog["weight_range"]
            weight = st.number_input("Weight (kg)", lo, hi, 2.04, 0.01)
        with st.expander("Display", icon=":material/desktop_windows:"):
            inches = choose("Diagonal (inches)", "Inches", 15.6)
            resolution = choose("Resolution", "Resolution", "1920x1080")
            ips = st.toggle("IPS panel", value=True)
            touch = st.toggle("Touchscreen")
    a, b = map(int, resolution.split("x"))
    return dict(Company=brand, TypeName=chassis, OpSys=os_name, Weight=weight,
                Cpu_Model=cpu, Gpu_Model=gpu, Gpu_Brand=catalog["gpus"][gpu],
                **catalog["cpus"][cpu], Ram=ram, SSD_GB=ssd, HDD_GB=hdd, Inches=inches,
                X_res=a, Y_res=b, PPI=round(math.hypot(a, b) / inches, 3),
                IPS=int(ips), Touchscreen=int(touch))


@st.cache_data(max_entries=48, show_spinner=False)
def render_waterfall(baseline, price, totals, labels, limit):
    return waterfall_png(baseline, price, totals, labels, limit)


def show_explanation(price, baseline, totals, labels, catalog):
    reference = catalog["audit"]["reference"]
    with st.container(border=True):
        with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
            st.header("Price breakdown", anchor=False)
            view = st.segmented_control(
                "Explanation view", ["This laptop", "Across laptops"], default="This laptop",
                required=True, label_visibility="collapsed",
            )
        if view == "Across laptops":
            st.caption(f"Average absolute SHAP contribution across {catalog['test_rows']} held-out "
                       "laptops. The bars show how much each feature moved prices in either direction.")
            ranking = pd.DataFrame([{"Feature": name, "Average impact (€)": value["mean_abs"]}
                                    for name, value in reference.items()]).sort_values("Average impact (€)", ascending=False)
            st.bar_chart(ranking, x="Feature", y="Average impact (€)", horizontal=True,
                         sort=False, color=POSITIVE, height=480,
                         alt="Features ranked by their average absolute contribution to held-out prices.")
        else:
            with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
                st.caption("SHAP waterfall · ← Lowers the estimate · Raises the estimate →")
                full = st.toggle("Show every feature", value=False, key="waterfall_all")
            largest = f"{totals.index[0]} at {signed_euros(totals.iloc[0])}"
            with st.container(horizontal=True, horizontal_alignment="center"):
                st.image(render_waterfall(baseline, price, totals, labels, len(totals) if full else 7),
                         width=980, alt=f"SHAP waterfall from €{baseline:,.2f} to €{price:,.2f}. "
                         f"The largest absolute contribution is {largest}.")


def show_feature_details(detail, totals, labels, reference):
    with st.expander("How does SHAP assign the euros?"):
        st.write("Imagine revealing the specs one at a time. The change attributed to RAM can "
                 "depend on whether the processor and storage are already known. TreeSHAP "
                 "averages these changes over possible feature orders using the fitted trees "
                 "and their training-path statistics. We sum the indicators for each feature, "
                 "so all brand indicators appear together as Brand. The baseline plus every "
                 "feature contribution recovers the prediction, up to rounding.")
    field = st.selectbox("Inspect a feature", totals.index.tolist())
    impact = totals[field]
    stats = reference[field]
    st.write(f"**{field} · {labels[field]}** · **{signed_euros(impact)}** in this estimate.")
    st.caption(f"Middle 80% of held-out contributions for {field.lower()} · "
               f"{signed_euros(stats['low'])} to {signed_euros(stats['high'])}")
    with st.expander("See the underlying model inputs"):
        subset = detail.loc[detail.Feature.eq(field), ["Indicator", "Input", "SHAP (€)"]].copy()
        if subset.Indicator.ne("").any():
            st.caption("Categories have one indicator per learned option. An indicator is 1 when present "
                       "and 0 when absent. Both states can contribute. All rows sum to the feature total.")
        subset["Indicator"] = subset.Indicator.replace("", field)
        st.dataframe(subset.sort_values("SHAP (€)", key=abs, ascending=False), hide_index=True,
                     column_config={"SHAP (€)": st.column_config.NumberColumn(format="%.3f")})
    st.caption("Changing a spec can also change other features’ contributions. To measure a specific "
               "upgrade, compare the two full predictions.")


def show_model_error(catalog):
    errors = catalog["audit"].get("absolute_errors_eur")
    if not errors:
        return
    frame = pd.DataFrame({
        "Absolute error (€)": [0.0, *errors],
        "Share of test laptops": [i / len(errors) for i in range(len(errors) + 1)],
    }).groupby("Absolute error (€)", as_index=False).max()
    st.subheader("How far off were the test predictions?", anchor=False)
    st.caption(f"About 90% of the {len(errors)} held-out catalog predictions were within "
               f"€{catalog['audit']['p90_error']:,.0f} of the recorded price. "
               "This curve measures historical catalog errors. The current listing has its own condition and market.")
    st.vega_lite_chart(frame, {
        "mark": {"type": "line", "interpolate": "step-after", "color": POSITIVE, "strokeWidth": 2.5},
        "height": 220,
        "encoding": {
            "x": {"field": "Absolute error (€)", "type": "quantitative"},
            "y": {"field": "Share of test laptops", "type": "quantitative",
                  "axis": {"format": ".0%"}, "scale": {"domain": [0, 1]}},
            "tooltip": [{"field": "Absolute error (€)", "type": "quantitative", "format": ",.2f"},
                        {"field": "Share of test laptops", "type": "quantitative", "format": ".0%"}],
        },
    }, width="stretch", alt="Cumulative share of held-out catalog laptops at each absolute euro error threshold.")


def main():
    st.set_page_config(page_title="SpecWorth", page_icon=":material/laptop_windows:", layout="wide")
    model, catalog = cached_bundle()
    row = configure(catalog)
    price, baseline, _, aligned, detail = explain(model, catalog, row)
    totals = ranked_features(detail)
    labels = feature_values(row, detail)
    st.title("SpecWorth", anchor=False)
    st.caption(f"Used laptop valuation · {row['Company']} {row['TypeName'].lower()} · "
               f"{row['Ram']} GB RAM · {row['Cpu_Model']}")
    price_card, reference, error = st.columns([1.5, 1, 1])
    price_card.metric("Estimated catalog price", f"€{price:,.2f}",
                      f"{signed_euros(price - baseline)} vs. baseline", delta_color="off",
                      delta_arrow="off", border=True, height="stretch")
    reference.metric("Model baseline", f"€{baseline:,.2f}", border=True, height="stretch",
                     help="The starting value for the model’s feature contributions.")
    error.metric("Average test error", f"€{catalog['metrics']['mae']:,.0f}", border=True,
                 height="stretch", help="Mean absolute error on held-out catalog configurations. Individual errors vary.")
    listing = listing_inputs()
    show_explanation(price, baseline, totals, labels, catalog)
    show_report(row, price, baseline, totals, labels, catalog, listing)
    show_feature_details(detail, totals, labels, catalog["audit"]["reference"])
    with st.expander("Model and evaluation"):
        st.write(f"The catalog contains {catalog['rows']:,} laptops. We held out {catalog['test_rows']} "
                 "rows and kept configurations with matching broad hardware specs in the same split. "
                 f"Holdout R² was {catalog['metrics']['r2']:.3f}. Prices reflect this catalog.")
        audit = catalog["audit"]
        st.write(f"Repeated grouped cross-validation averaged €{audit['cv_mae']:,.0f} absolute error. "
                 f"Holding out named products in a separate development-data check averaged "
                 f"€{audit['product_mae']:,.0f}. The 90th percentile absolute error on the main holdout "
                 f"was €{audit['p90_error']:,.0f}.")
        show_model_error(catalog)
        st.dataframe(pd.DataFrame(audit["bands"]).rename(columns={
            "band": "Actual catalog price", "rows": "Laptops", "mae": "Mean absolute error (€)"
        }).round(2), hide_index=True)
        st.caption(f"Encoded model inputs · {aligned.shape[1]} columns")
        st.dataframe(aligned, hide_index=True, width="stretch")


if __name__ == "__main__":
    main()

