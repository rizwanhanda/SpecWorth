"""Streamlit controls and pricing strategy report for a used listing."""
import json

import streamlit as st
from google import genai
from google.genai import errors, types
from httpx2 import RequestError
from streamlit.errors import StreamlitSecretNotFoundError

from explanations import signed_euros
from valuation import CONDITIONS, WORKLOADS, hardware_fit, market_assessment, purchase_guidance

GEMINI_MODEL = "gemini-3.5-flash-lite"
SECTION_TITLES = {
    "valuation": "Price analysis",
    "hardware": "Fit and trade-offs",
    "strategy": "Negotiation and purchase checks",
}


def listing_inputs():
    with st.container(border=True):
        st.subheader("Evaluate a used listing", anchor=False)
        price_field, condition_field, workload_field = st.columns([1, 1.4, 1.7])
        ask = price_field.number_input("Asking price per laptop (€)", min_value=.01, step=10.0,
                                       value=None, format="%.2f", placeholder="Enter the seller’s price")
        condition = condition_field.selectbox("Seller-reported condition", CONDITIONS)
        workload = workload_field.selectbox("What will you use it for?", WORKLOADS)
        with st.expander("Add comparable used prices"):
            st.caption("Use three recent examples from the same market with similar specs and condition. "
                       "Compare total prices in euros, including required fees and shipping.")
            source = st.radio("Price source", ["Completed sale prices", "Asking prices"], horizontal=True)
            comparisons = [
                column.number_input(f"Comparable {number} (€)", min_value=.01, step=10.0, value=None,
                                    format="%.2f", placeholder="Enter a price")
                for number, column in enumerate(st.columns(3), start=1)
            ]
    return ask, condition, workload, [price for price in comparisons if price is not None], source


def show_report(row, price, baseline, totals, labels, catalog, listing):
    ask, condition, workload, comparisons, source = listing
    market = market_assessment(ask, comparisons, source)
    fit = hardware_fit(row, workload)
    purchase_action, checks = purchase_guidance(condition, fit)
    pricing_next_step = market.next_step if fit.suitable is not False and condition != CONDITIONS[3] else ""
    shifts = totals[totals.map(signed_euros).ne("€0.00")]
    higher = shifts[shifts.gt(0)].head(2)
    lower = shifts[shifts.lt(0)].head(2)

    with st.container(border=True):
        st.header("Pricing strategy report", anchor=False)
        st.subheader(market.verdict, anchor=False)
        if ask is not None:
            st.markdown(f"**€{ask:,.2f} asking price per laptop** · {condition}")
        st.write(market.reason)
        st.write(f"**{fit.title}.** You selected {workload.lower()}. {fit.reason}")

        st.subheader("What drives the model price", anchor=False)
        st.markdown(f"The retail catalog estimate is **€{price:,.2f}** "
                    f"({signed_euros(price - baseline)} versus its €{baseline:,.2f} model baseline).")
        if ask is not None:
            st.caption(f"The asking price differs from that catalog estimate by {signed_euros(ask - price)} "
                       "(asking price minus estimate). Use your used-price comparisons for the deal verdict.")
        raises, lowers = st.columns(2)
        for column, title, drivers in (
            (raises, "Raised the estimate", higher),
            (lowers, "Lowered the estimate", lower),
        ):
            with column:
                st.markdown(f"**{title}**")
                if drivers.empty:
                    st.caption("No contributions to the nearest cent.")
                else:
                    st.markdown("\n".join(
                        f"- {name} · {labels[name]} · **{signed_euros(amount)}**"
                        for name, amount in drivers.items()
                    ))
        st.caption(f"Across {catalog['test_rows']} held-out catalog laptops, estimates missed by "
                   f"**€{catalog['metrics']['mae']:,.0f} on average**. The model learned retail catalog "
                   "prices. Condition, battery wear, and today’s resale prices need separate evidence.")

        st.subheader("How to proceed", anchor=False)
        st.write(f"**{purchase_action}.** {checks}")
        if pricing_next_step:
            st.write(pricing_next_step)
        show_gemini_report({
            "selected_specs": labels, "intended_use": workload, "seller_reported_condition": condition,
            "market_assessment": market.verdict, "price_evidence": market.reason,
            "hardware_fit": fit.title, "hardware_details": fit.reason,
            "catalog_estimate_eur": round(price, 2), "model_baseline_eur": round(baseline, 2),
            "largest_contributions_eur": {name: round(amount, 2) for name, amount in (*higher.items(), *lower.items())},
            "purchase_action": purchase_action, "pricing_next_step": pricing_next_step, "condition_checks": checks,
        })


def generate_strategy(key, facts):
    prompt = (
        "Write a professional pricing strategy report for a buyer of used laptops. Return three short paragraphs "
        "with the requested keys: valuation, hardware, strategy. Aim for two or three sentences per paragraph. "
        "Explain how the asking price and available comparisons support the verdict. Relate the important "
        "feature contributions to the historical catalog estimate. Explain fit for the named workload, "
        "meaningful trade-offs, and exactly what evidence to request or use in negotiation. "
        "Preserve the supplied price verdict, hardware fit and purchase action. If the purchase action "
        "advises a hold or pass, lead with that action. Keep measured facts "
        "and unverified seller claims distinct. Use only the supplied facts. All figures are displayed "
        "by the app, so refer to them in words and use no digits, new prices, percentages, or price targets. "
        "Do not make claims about current market prices, resale value, performance benchmarks, or a verified "
        "condition. Do not describe feature contributions as upgrade costs or discounts.\n"
    )
    schema = {
        "type": "object",
        "properties": {name: {"type": "string"} for name in SECTION_TITLES},
        "required": list(SECTION_TITLES),
    }
    config = types.GenerateContentConfig(response_mime_type="application/json", response_json_schema=schema)
    with genai.Client(api_key=key, http_options=types.HttpOptions(timeout=25000)) as client:
        result = client.models.generate_content(model=GEMINI_MODEL, contents=prompt + json.dumps(facts), config=config)
    try:
        report = json.loads(result.text or "")
    except ValueError:
        return None
    if not isinstance(report, dict) or set(report) != set(SECTION_TITLES):
        return None
    for text in report.values():
        if (not isinstance(text, str) or not 15 <= len(text.split()) <= 95
                or any(char.isdigit() or char in "€$£%" for char in text)):
            return None
    return report


def show_gemini_report(facts):
    try:
        key = st.secrets.get("GEMINI_API_KEY", "").strip()
    except StreamlitSecretNotFoundError:
        return
    if not key:
        return

    signature = json.dumps(facts, sort_keys=True)
    if st.session_state.get("strategy_signature") != signature:
        st.session_state.pop("strategy_report", None)
        st.session_state["strategy_signature"] = signature
    if st.button("Generate detailed buyer analysis", type="primary",
                 help="Sends this listing’s entered specs and valuation evidence to Google Gemini."):
        try:
            with st.spinner("Analyzing price, fit, and negotiation options…"):
                report = generate_strategy(key, facts)
            if report:
                st.session_state["strategy_report"] = report
            else:
                st.warning("The detailed analysis did not pass its output checks. The valuation above is still available.")
        except errors.APIError as error:
            if error.code == 429:
                st.warning("Gemini’s request limit was reached. Try again later.")
            else:
                st.error("Gemini could not complete the analysis. Check key and model access.")
        except (RequestError, TimeoutError):
            st.warning("Google could not be reached. Try again in a moment.")
    if report := st.session_state.get("strategy_report"):
        st.caption("Gemini analysis based on the report evidence above")
        for section, name in SECTION_TITLES.items():
            st.markdown(f"**{name}**")
            st.text(report[section])
