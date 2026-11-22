"""Streamlit entry point for the laptop catalog price estimator."""
import json
import math

import pandas as pd
import streamlit as st
from google import genai
from google.genai import errors, types
from httpx2 import RequestError
from streamlit.errors import StreamlitSecretNotFoundError

from explanations import POSITIVE, feature_values, ranked_features, signed_euros, waterfall_png
from pricing import explain, load_bundle

GEMINI_MODEL = "gemini-3.5-flash-lite"
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


def generate_takeaway(key, specs):
    prompt = ("Write one plain-English sentence of at most 30 words suggesting what a laptop "
              "buyer should compare in current listings. Use only the supplied specs. "
              "Avoid price figures, claims about performance or value, and buy or skip advice. "
              "Return just the sentence.\n")
    with genai.Client(api_key=key, http_options=types.HttpOptions(timeout=20000)) as client:
        result = client.models.generate_content(model=GEMINI_MODEL, contents=prompt + json.dumps(specs))
    text = (result.text or "").strip()
    if "\n" not in text and len(text.split()) <= 30:
        return text
    return ""


def briefing(row, price, baseline, totals, labels, catalog):
    difference = round(price - baseline, 2)
    if difference:
        direction = "above" if difference > 0 else "below"
        comparison = f"€{abs(difference):,.2f} {direction} the €{baseline:,.2f} model baseline."
    else:
        comparison = f"Matches the €{baseline:,.2f} model baseline."
    displayed = totals[totals.map(signed_euros).ne("€0.00")]
    with st.container(border=True):
        st.subheader("Buyer’s briefing", anchor=False)
        st.markdown(f"**€{price:,.2f} estimated catalog price** · {comparison}")
        raised, lowered = st.columns(2)
        for column, heading, drivers in (
            (raised, "Raised the estimate", displayed[displayed.gt(0)]),
            (lowered, "Lowered the estimate", displayed[displayed.lt(0)]),
        ):
            with column:
                st.markdown(f"**{heading}**")
                if drivers.empty:
                    st.caption("No contributions to the nearest cent.")
                else:
                    st.markdown("\n".join(
                        f"- {name} · {labels[name]} · **{signed_euros(amount)}**"
                        for name, amount in drivers.head(2).items()
                    ))
        st.markdown("**How to use it**")
        st.write(f"The model was off by about **€{catalog['metrics']['mae']:,.0f} on average** across "
                 f"{catalog['test_rows']} test laptops. Use the estimate as a starting point, "
                 "then compare current listings with the same processor, memory, and storage.")
        show_optional_takeaway(row, labels)


def show_optional_takeaway(row, labels):
    try:
        key = st.secrets.get("GEMINI_API_KEY", "").strip()
    except StreamlitSecretNotFoundError:
        key = ""
    if not key:
        return
    signature = json.dumps(row, sort_keys=True)
    if st.session_state.get("takeaway_signature") != signature:
        st.session_state.pop("takeaway_text", None)
        st.session_state["takeaway_signature"] = signature
    if st.button("Get a comparison tip with Gemini", help="Sends the selected laptop specs to Google."):
        try:
            with st.spinner("Writing comparison tip…"):
                response = generate_takeaway(key, labels)
            if response:
                st.session_state["takeaway_text"] = response
            else:
                st.warning("Gemini did not return a short comparison tip. Try again.")
        except errors.APIError as error:
            if error.code == 429:
                st.warning("Gemini’s request limit was reached. Try again later.")
            else:
                st.error("Gemini could not complete the request. Check the API key and model access.")
        except (RequestError, TimeoutError):
            st.warning("Google could not be reached. Try again in a moment.")
    if text := st.session_state.get("takeaway_text"):
        st.text(text)


def main():
    st.set_page_config(page_title="SpecWorth", page_icon=":material/laptop_windows:", layout="wide")
    model, catalog = cached_bundle()
    row = configure(catalog)
    price, baseline, _, aligned, detail = explain(model, catalog, row)
    totals = ranked_features(detail)
    labels = feature_values(row, detail)
    st.title("SpecWorth", anchor=False)
    st.caption(f"Laptop price estimator · {row['Company']} {row['TypeName'].lower()} · "
               f"{row['Ram']} GB RAM · {row['Cpu_Model']}")
    price_card, reference, error = st.columns([1.5, 1, 1])
    price_card.metric("Estimated catalog price", f"€{price:,.2f}",
                      f"{signed_euros(price - baseline)} vs. baseline", delta_color="off",
                      delta_arrow="off", border=True, height="stretch")
    reference.metric("Model baseline", f"€{baseline:,.2f}", border=True, height="stretch",
                     help="The starting value for the model’s feature contributions.")
    error.metric("Average test error", f"€{catalog['metrics']['mae']:,.0f}", border=True,
                 height="stretch", help="Mean absolute error on held-out catalog configurations. Individual errors vary.")
    show_explanation(price, baseline, totals, labels, catalog)
    briefing(row, price, baseline, totals, labels, catalog)
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
        st.dataframe(pd.DataFrame(audit["bands"]).rename(columns={
            "band": "Actual catalog price", "rows": "Laptops", "mae": "Mean absolute error (€)"
        }).round(2), hide_index=True)
        st.caption(f"Encoded model inputs · {aligned.shape[1]} columns")
        st.dataframe(aligned, hide_index=True, width="stretch")


if __name__ == "__main__":
    main()

