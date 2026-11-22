"""Streamlit entry point for the laptop catalog price estimator."""
import json
import math

import pandas as pd
import streamlit as st
from google import genai
from google.genai import errors, types
from httpx2 import RequestError
from streamlit.errors import StreamlitSecretNotFoundError

from explanations import feature_values, ranked_features, waterfall_png
from pricing import explain, load_bundle

GEMINI_MODEL = "gemini-3.5-flash-lite"
cached_bundle = st.cache_resource(load_bundle)


def configure(catalog):
    choices = catalog["choices"]

    def choose(label, field, default):
        options = choices[field]
        return st.selectbox(label, options, index=options.index(default))

    with st.sidebar:
        st.header("Configuration")
        st.subheader("Device")
        brand = choose("Brand", "Company", "Dell")
        chassis = choose("Form factor", "TypeName", "Notebook")
        os_name = choose("Operating system", "OpSys", "Windows 10")
        lo, hi = catalog["weight_range"]
        weight = st.number_input("Weight (kg)", lo, hi, 2.04, 0.01)
        st.subheader("Hardware")
        cpus, gpus = sorted(catalog["cpus"]), sorted(catalog["gpus"])
        cpu = st.selectbox("Processor", cpus, index=cpus.index("Intel Core i5 7200U 2.5GHz"))
        gpu = st.selectbox("Graphics", gpus, index=gpus.index("Intel HD Graphics 620"))
        ram = choose("Memory (GB)", "Ram", 8)
        ssd = choose("SSD or flash storage (GB)", "SSD_GB", 256)
        hdd = choose("HDD or hybrid storage (GB)", "HDD_GB", 0)
        st.subheader("Display")
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


def show_explanation(row, price, baseline, detail, catalog):
    st.header("Inside the estimate")
    st.write("Start at the model baseline and follow the bars to the price. Teal adds euros. "
             "Copper subtracts them. Each amount is specific to the selected laptop.")
    with st.expander("How does SHAP assign the euros?"):
        st.write("Imagine revealing the specs one at a time. The change attributed to RAM can "
                 "depend on whether the processor and storage are already known. TreeSHAP "
                 "averages these changes over possible feature orders using the fitted trees "
                 "and their training-path statistics. We sum the indicators for each feature, "
                 "so all brand indicators appear together as Brand. The baseline plus every "
                 "feature contribution recovers the prediction, up to rounding.")
    totals = ranked_features(detail)
    labels = feature_values(row, detail)
    reference = catalog["audit"]["reference"]
    view = st.segmented_control("Explanation view", ["This laptop", "Across laptops"],
                                default="This laptop", selection_mode="single")
    if view == "Across laptops":
        st.subheader("Which features usually move the price?")
        st.caption(f"Mean absolute SHAP contribution across the {catalog['test_rows']} held-out laptops. "
                   "Larger amounts indicate a stronger average influence in either direction.")
        ranking = pd.DataFrame([{"Feature": name, "Average impact (€)": value["mean_abs"]}
                                for name, value in reference.items()]).sort_values("Average impact (€)", ascending=False)
        st.bar_chart(ranking, x="Feature", y="Average impact (€)", horizontal=True,
                     sort=False, color="#087b70", height=480,
                     alt="Features ranked by their average absolute contribution to held-out prices.")
    else:
        st.subheader("From baseline to this laptop")
        full = st.toggle("Show every feature", value=False)
        st.image(render_waterfall(baseline, price, totals, labels, len(totals) if full else 7),
                 width="stretch", alt=f"SHAP waterfall from €{baseline:,.2f} to €{price:,.2f}. "
                 f"The largest absolute contribution is {totals.index[0]} at {totals.iloc[0]:+,.2f} euros.")
    field = st.selectbox("Inspect a feature", totals.index.tolist())
    impact = totals[field]
    stats = reference[field]
    st.write(f"**{field} · {labels[field]}** contributes **{impact:+,.2f} €** to this prediction. "
             f"The middle 80% of held-out laptops had contributions for this feature between "
             f"€{stats['low']:,.2f} and €{stats['high']:,.2f}.")
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


def generate_briefing(key, facts):
    prompt = ("Write two short plain-text paragraphs for a laptop buyer. Explain the catalog "
              "estimate using the supplied figures and largest grouped contributions. "
              "Keep the figures in euros. These are model associations. Avoid claims about "
              "upgrade costs, current retail prices, performance benchmarks or discounts.\n")
    with genai.Client(api_key=key, http_options=types.HttpOptions(timeout=20000)) as client:
        result = client.models.generate_content(model=GEMINI_MODEL, contents=prompt + json.dumps(facts))
    return (result.text or "").strip()


def briefing(row, price, baseline, impacts):
    st.subheader("Buyer’s briefing")
    st.caption("Generate a short explanation with Gemini. This sends the selected specs and model results to Google.")
    try:
        key = st.secrets.get("GEMINI_API_KEY", "").strip()
    except StreamlitSecretNotFoundError:
        key = ""
    if not key:
        st.info("Set GEMINI_API_KEY in Streamlit secrets to enable the briefing.")
        return
    facts = dict(configuration=row, estimate_eur=round(price, 2), baseline_eur=round(baseline, 2),
                 contributions_eur={name: round(value, 2) for name, value in impacts.items()})
    signature = json.dumps(facts, sort_keys=True)
    if st.session_state.get("briefing_signature") != signature:
        st.session_state.pop("briefing_text", None)
        st.session_state["briefing_signature"] = signature
    if st.button("Generate briefing", type="primary"):
        try:
            with st.spinner("Writing briefing…"):
                response = generate_briefing(key, facts)
            if response:
                st.session_state["briefing_text"] = response
            else:
                st.warning("Gemini returned an empty briefing. Try again.")
        except errors.APIError as error:
            if error.code == 429:
                st.warning("Gemini’s request limit was reached. Try again later.")
            else:
                st.error("Gemini could not complete the request. Check the API key and model access.")
        except (RequestError, TimeoutError):
            st.warning("Google could not be reached. Try again in a moment.")
    if text := st.session_state.get("briefing_text"):
        st.text(text)


def main():
    st.set_page_config(page_title="Laptop price estimator", page_icon="💻", layout="wide")
    model, catalog = cached_bundle()
    row = configure(catalog)
    price, baseline, impacts, aligned, detail = explain(model, catalog, row)
    st.title("Laptop price estimator")
    st.caption(f"{row['Company']} {row['TypeName'].lower()} · {row['Cpu_Model']} · "
               f"{row['Ram']} GB RAM · {row['Gpu_Model']}")
    price_card, reference, error = st.columns(3)
    price_card.metric("Estimated catalog price", f"€{price:,.2f}", f"{price-baseline:+,.2f} € vs baseline", border=True)
    reference.metric("Model baseline", f"€{baseline:,.2f}", border=True,
                     help="The starting value for the model’s feature contributions.")
    error.metric("Average test error", f"€{catalog['metrics']['mae']:,.0f}", border=True,
                 help="Mean absolute error on held-out catalog configurations. Individual errors vary.")
    show_explanation(row, price, baseline, detail, catalog)
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
    st.divider()
    briefing(row, price, baseline, impacts)


if __name__ == "__main__":
    main()

