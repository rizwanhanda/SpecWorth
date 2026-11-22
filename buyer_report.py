"""Streamlit controls and pricing strategy report for a used listing."""
import json
import logging
import re

import streamlit as st
from google import genai
from google.genai import errors, types
from httpx2 import RequestError
from streamlit.errors import StreamlitSecretNotFoundError

from explanations import signed_euros
from valuation import CONDITIONS, WORKLOADS, hardware_fit, market_assessment, purchase_guidance

GEMINI_MODEL = "gemini-3.5-flash-lite"
logger = logging.getLogger(__name__)
SECTION_TITLES = {
    "valuation": "Price analysis",
    "hardware": "Fit and trade-offs",
    "strategy": "Negotiation and purchase checks",
}
GEMINI_SECTIONS = ("hardware", "strategy")
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
PRICE_LANGUAGE = re.compile(r"[%€$£₹]|\b(?:EUR|USD|GBP|INR|euros?|dollars?|pounds?|percent)\b", re.IGNORECASE)


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
            "asking_price_eur": ask, "comparable_count": len(comparisons),
            "market_assessment": market.verdict, "price_evidence": market.reason,
            "hardware_fit": fit.title, "hardware_details": fit.reason,
            "catalog_estimate_eur": round(price, 2), "model_baseline_eur": round(baseline, 2),
            "largest_contributions_eur": {name: round(amount, 2) for name, amount in (*higher.items(), *lower.items())},
            "purchase_action": purchase_action, "pricing_next_step": pricing_next_step, "condition_checks": checks,
        })


def local_strategy(facts):
    ask = facts["asking_price_eur"]
    opening = f"{facts['market_assessment']}."
    if ask is not None:
        opening += f" The seller is asking €{ask:,.2f} per laptop."
    price_evidence = facts["price_evidence"]
    catalog_estimate = facts["catalog_estimate_eur"]
    valuation = (f"{opening} {price_evidence} The €{catalog_estimate:,.2f} catalog estimate is a historical "
                 "retail benchmark. Use recent used sales to judge the offer.")
    return {
        "valuation": valuation,
        "hardware": f"{facts['hardware_fit']}. {facts['hardware_details']}",
        "strategy": " ".join(filter(None, (f"{facts['purchase_action']}.", facts["condition_checks"],
                                           facts["pricing_next_step"]))),
    }


def generate_strategy(key, facts):
    prompt = (
        "Write two short professional paragraphs to supplement an app-generated used laptop valuation. "
        "Return the requested keys hardware and strategy. In hardware, explain the fit for the intended use "
        "and the most important trade-off using the supplied specs. In strategy, explain what evidence to "
        "request or what to negotiate. Follow the supplied purchase action and condition checks. If fewer "
        "than three comparable prices are available, explain what comparable evidence is needed and avoid "
        "implying that the asking price is fair or expensive. The app displays all prices and the overall "
        "verdict. Do not repeat monetary amounts, invent results, give performance benchmarks, or present "
        "seller claims as verified. Hardware numbers are fine if copied from the supplied specs.\n"
    )
    schema = {
        "type": "object",
        "properties": {name: {"type": "string"} for name in GEMINI_SECTIONS},
        "required": list(GEMINI_SECTIONS),
    }
    config = types.GenerateContentConfig(response_mime_type="application/json", response_json_schema=schema)
    retry = types.HttpRetryOptions(attempts=2, initial_delay=.5, max_delay=1, jitter=.25,
                                   http_status_codes=[408, 500, 502, 503, 504])
    with genai.Client(api_key=key, http_options=types.HttpOptions(timeout=15000, retry_options=retry)) as client:
        result = client.models.generate_content(model=GEMINI_MODEL, contents=prompt + json.dumps(facts), config=config)
    try:
        report = json.loads(result.text or "")
    except ValueError:
        return {}
    if not isinstance(report, dict):
        return {}
    allowed_numbers = {
        number for value in facts["selected_specs"].values() for number in NUMBER.findall(value)
    }
    accepted = {}
    for section in GEMINI_SECTIONS:
        text = report.get(section)
        if not isinstance(text, str):
            continue
        text = " ".join(text.split())
        if 8 <= len(text.split()) <= 120 and not PRICE_LANGUAGE.search(text):
            if set(NUMBER.findall(text)) <= allowed_numbers:
                accepted[section] = text
    return accepted


def api_status_message(code):
    if code in (401, 403):
        return "Google denied API access"
    if code == 402:
        return "Google reported a billing issue"
    if code == 404:
        return "Google could not find the configured model"
    if code == 429:
        return "Gemini’s request limit was reached"
    if code in (408, 504):
        return "Gemini timed out"
    if 500 <= code < 600:
        return "Gemini is temporarily unavailable"
    return "Google rejected the Gemini request"


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
        st.session_state.pop("strategy_note", None)
        st.session_state["strategy_signature"] = signature
    if st.button("Generate detailed buyer analysis", type="primary",
                 help="Sends this listing’s entered specs and valuation evidence to Google Gemini."):
        report = local_strategy(facts)
        note = "Detailed analysis · Price verdict from your listing evidence"
        try:
            with st.spinner("Analyzing price, fit, and negotiation options…"):
                prose = generate_strategy(key, facts)
            report.update(prose)
            if len(prose) < len(GEMINI_SECTIONS):
                logger.info("Gemini strategy completed %d of %d sections", len(prose), len(GEMINI_SECTIONS))
                note = ("Some Gemini sections were unavailable. The report includes analysis from your listing."
                        if prose else "Gemini’s reply could not be used. Showing analysis from your listing.")
        except errors.APIError as error:
            logger.warning("Gemini strategy request failed with HTTP %s", error.code)
            note = f"{api_status_message(error.code)} (HTTP {error.code}). Showing analysis from your listing."
        except (RequestError, TimeoutError):
            logger.warning("Gemini strategy request timed out or could not reach Google")
            note = "Gemini could not be reached in time. Showing analysis from your listing."
        st.session_state["strategy_report"] = report
        st.session_state["strategy_note"] = note
    if report := st.session_state.get("strategy_report"):
        st.caption(st.session_state.get("strategy_note", "Detailed buyer analysis"))
        for section, name in SECTION_TITLES.items():
            st.markdown(f"**{name}**")
            st.text(report[section])
