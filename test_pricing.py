"""Offline regression tests for the shipped bundle and explanations."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx2
from google.genai import errors
from matplotlib.figure import Figure

from app import configure, main, st
from buyer_report import generate_strategy, show_gemini_report, show_report
from explanations import draw_step, feature_values, ranked_features, waterfall_png
from pricing import explain, load_bundle
from valuation import CONDITIONS, WORKLOADS, hardware_fit, market_assessment, purchase_guidance


class PricingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.catalog = load_bundle()
        cls.row = configure(cls.catalog)

    def test_every_feature_reconciles(self):
        for ram in self.catalog["choices"]["Ram"]:
            row = dict(self.row, Ram=ram)
            price, baseline, groups, inputs, detail = explain(self.model, self.catalog, row)
            totals = ranked_features(detail)
            self.assertEqual(len(detail), inputs.shape[1])
            self.assertEqual(set(feature_values(row, detail)), set(totals.index))
            self.assertEqual(set(totals.index), set(self.catalog["audit"]["reference"]))
            self.assertAlmostEqual(price, baseline + totals.sum(), delta=.02)
            self.assertAlmostEqual(price, baseline + sum(groups.values()), delta=.02)

    def test_one_hot_explanation_includes_absent_options(self):
        _, _, _, _, detail = explain(self.model, self.catalog, self.row)
        brand = detail.loc[detail.Feature.eq("Brand")]
        self.assertEqual(brand.loc[brand.Input.eq(1), "Indicator"].tolist(), [self.row["Company"]])
        self.assertTrue((brand.loc[brand.Input.eq(0), "SHAP (€)"].abs() > 0).any())

    def test_both_waterfalls_render(self):
        price, base, _, _, detail = explain(self.model, self.catalog, self.row)
        totals, labels = ranked_features(detail), feature_values(self.row, detail)
        for limit in (7, len(totals)):
            image = waterfall_png(base, price, totals, labels, limit)
            self.assertTrue(image.startswith(b"\x89PNG"))

    def test_arrows_end_at_exact_running_price(self):
        axis = Figure().subplots()
        for start, end in [(100, 110), (110, 110.02), (110.02, 98), (98, 97.99)]:
            with self.subTest(start=start, end=end):
                draw_step(axis, start, end, 2, "blue", max_head=2)
                vertices = axis.patches[-1].get_xy()
                self.assertEqual(tuple(vertices[3]), (end, 2))
                self.assertEqual(vertices[:, 0].min(), min(start, end))
                self.assertEqual(vertices[:, 0].max(), max(start, end))
        draw_step(axis, 100, 100, 2, "blue", max_head=2)
        self.assertEqual(len(axis.patches), 4)
        self.assertEqual(axis.lines[-1].get_marker(), "|")

    def test_app_views_without_credentials(self):
        with patch.object(st, "secrets", {}), patch.object(st, "segmented_control",
                side_effect=["This laptop", "Across laptops"]), patch.object(st, "header") as heading, \
                patch.object(st, "button") as button:
            main()
            main()
            self.assertEqual(sum(call.args[0] == "Pricing strategy report" for call in heading.call_args_list), 2)
            button.assert_not_called()

    def test_used_report_matches_the_waterfall_and_comparables_without_gemini(self):
        row = dict(self.row, Weight=2.05)
        price, baseline, _, _, detail = explain(self.model, self.catalog, row)
        listing = (300.0, CONDITIONS[1], WORKLOADS[0], [275.0, 310.0, 340.0], "Completed sale prices")
        with patch.object(st, "secrets", {}), patch.object(st, "markdown") as markdown, \
                patch.object(st, "write") as write, patch.object(st, "subheader") as subheader:
            show_report(row, price, baseline, ranked_features(detail), feature_values(row, detail), self.catalog, listing)
        self.assertIn("Competitive against your comparisons", [call.args[0] for call in subheader.call_args_list])
        content = "\n".join(call.args[0] for call in markdown.call_args_list)
        for expected in ("€300.00 asking price", "€1,027.49", "€1,108.02", "Solid-state storage · 256 GB · **+€48.22**",
                         "IPS panel · Yes · **+€33.86**", "Weight · 2.05 kg · **−€72.28**",
                         "Form factor · Notebook · **−€62.42**"):
            self.assertIn(expected, content)
        advice = "\n".join(call.args[0] for call in write.call_args_list)
        self.assertIn("The median is €310.00.", advice)
        self.assertIn("Suitable on paper for everyday work", advice)

    def test_market_verdict_is_based_on_comparable_used_prices(self):
        comparisons = [280, 330, 300]
        cases = [(None, comparisons, "Enter"), (100, comparisons[:2], "Used-market"),
                 (240, comparisons, "Potentially good"), (280, comparisons, "Competitive"),
                 (300, comparisons, "Competitive"), (330, comparisons, "Reasonable"),
                 (360, comparisons, "Expensive")]
        for ask, prices, expected in cases:
            with self.subTest(ask=ask, prices=prices):
                result = market_assessment(ask, prices, "Completed sale prices")
                self.assertTrue(result.verdict.startswith(expected))
        report = market_assessment(360, comparisons, "Asking prices")
        self.assertIn("seller quotes", report.reason)
        self.assertIn("€300.00", report.next_step)

    def test_condition_and_workload_can_override_a_competitive_price(self):
        office = hardware_fit(self.row, WORKLOADS[0])
        gaming = hardware_fit(self.row, WORKLOADS[2])
        self.assertTrue(office.suitable)
        self.assertFalse(gaming.suitable)
        self.assertIn("better fit", purchase_guidance(CONDITIONS[1], gaming)[0])
        self.assertIn("inspection", purchase_guidance(CONDITIONS[0], office)[0])
        self.assertIn("battery", purchase_guidance(CONDITIONS[2], office)[0])
        self.assertIn("fault", purchase_guidance(CONDITIONS[3], gaming)[0])

    def test_historical_error_curve_has_all_held_out_residuals(self):
        errors = self.catalog["audit"]["absolute_errors_eur"]
        self.assertEqual(len(errors), self.catalog["test_rows"])
        self.assertEqual(errors, sorted(errors))
        self.assertAlmostEqual(sum(errors) / len(errors), self.catalog["metrics"]["mae"], places=4)
        threshold = self.catalog["audit"]["p90_error"]
        coverage = sum(error <= threshold for error in errors) / len(errors)
        self.assertAlmostEqual(coverage, .9, delta=1 / len(errors))

    def strategy_facts(self, ask=300):
        return {
            "asking_price_eur": ask,
            "comparable_count": 0,
            "selected_specs": {"Memory": "8 GB", "CPU model": "Intel Core i5 7200U 2.5GHz"},
            "market_assessment": "Used-market value needs more evidence",
            "price_evidence": "You entered 0 of the 3 comparable prices needed for a verdict.",
            "catalog_estimate_eur": 1037.73,
            "hardware_fit": "Suitable on paper for everyday work",
            "hardware_details": "8 GB is a reasonable starting point for light multitasking.",
            "purchase_action": "Factor in the worn battery",
            "condition_checks": "Ask for a battery report and replacement quote.",
            "pricing_next_step": "Find recent completed sales for the same specs and condition.",
        }

    def test_gemini_report_stays_tied_to_the_listing(self):
        body = {"hardware": "Hardware discussion", "strategy": "Purchase checks"}
        with patch.object(st, "secrets", {"GEMINI_API_KEY": "placeholder"}), \
             patch.object(st, "session_state", {}), patch.object(st, "button", side_effect=[True, False, False]), \
             patch("buyer_report.generate_strategy", return_value=body) as generate, patch.object(st, "text") as text:
            for ask in (300, 300, 310):
                show_gemini_report(self.strategy_facts(ask))
            generate.assert_called_once_with("placeholder", self.strategy_facts(300))
            self.assertEqual(text.call_count, 6)
            self.assertIn("€300.00", text.call_args_list[0].args[0])
            self.assertIn("0 of the 3 comparable prices", text.call_args_list[0].args[0])

    def test_gemini_keeps_valid_sections_with_exact_hardware_numbers(self):
        hardware = "The Intel Core i5 and 8 GB of memory are suitable for light multitasking."
        strategy = "Ask the seller for a battery report before agreeing to pay."
        report = {"hardware": hardware, "strategy": strategy}
        with patch("buyer_report.genai.Client") as sdk:
            send = sdk.return_value.__enter__.return_value.models.generate_content
            send.side_effect = [
                SimpleNamespace(text=json.dumps(dict(report, valuation="A price verdict from Gemini."))),
                SimpleNamespace(text=json.dumps(dict(report, strategy=strategy + " Offer €50."))),
                SimpleNamespace(text=json.dumps(dict(report, hardware=hardware.replace("8 GB", "32 GB")))),
                SimpleNamespace(text="Not JSON"),
            ]
            facts = self.strategy_facts()
            self.assertEqual(generate_strategy("placeholder", facts), report)
            self.assertEqual(generate_strategy("placeholder", facts), {"hardware": hardware})
            self.assertEqual(generate_strategy("placeholder", facts), {"strategy": strategy})
            self.assertEqual(generate_strategy("placeholder", facts), {})
            self.assertEqual(send.call_args.kwargs["config"].response_mime_type, "application/json")

    def test_guidance_survives_invalid_replies_and_api_errors(self):
        failures = [
            ({}, "reply could not be used"),
            ({"hardware": "Gemini hardware paragraph"}, "Some Gemini sections"),
            (errors.APIError(503, {"error": {"message": "test"}}), "temporarily unavailable (HTTP 503)"),
            (errors.APIError(429, {"error": {"message": "test"}}), "request limit was reached (HTTP 429)"),
            (errors.APIError(403, {"error": {"message": "test"}}), "denied API access (HTTP 403)"),
            (httpx2.ReadTimeout("test"), "could not be reached in time"),
        ]
        for outcome, note in failures:
            with self.subTest(note=note), patch.object(st, "secrets", {"GEMINI_API_KEY": "placeholder"}), \
                 patch.object(st, "session_state", {}), patch.object(st, "button", return_value=True), \
                 patch("buyer_report.generate_strategy", side_effect=[outcome]), \
                 patch.object(st, "caption") as caption, patch.object(st, "text") as text:
                show_gemini_report(self.strategy_facts())
                self.assertIn(note, caption.call_args.args[0])
                self.assertEqual(text.call_count, 3)
                self.assertIn("catalog estimate is a historical retail benchmark", text.call_args_list[0].args[0])
                self.assertIn("battery report", text.call_args_list[2].args[0])

    def test_sdk_retries_transient_errors_once_without_retrying_quota(self):
        hardware = "The 8 GB of memory is suitable for browsing and light multitasking."
        strategy = "Ask the seller for a battery report before agreeing to pay."
        success = {"candidates": [{"finishReason": "STOP", "content": {
            "role": "model", "parts": [{"text": json.dumps({"hardware": hardware, "strategy": strategy})}],
        }}]}
        for codes, expected_calls in [([503, 200], 2), ([503, 503], 2), ([429], 1)]:
            with self.subTest(codes=codes):
                remaining = iter(codes)

                def send(client, request, **kwargs):
                    code = next(remaining)
                    body = success if code == 200 else {"error": {"code": code, "message": "test"}}
                    return httpx2.Response(code, json=body, request=request)

                with patch.object(httpx2.Client, "send", autospec=True, side_effect=send) as transport:
                    if codes[-1] == 200:
                        self.assertEqual(generate_strategy("placeholder", self.strategy_facts())["hardware"], hardware)
                    else:
                        with self.assertRaises(errors.APIError) as failure:
                            generate_strategy("placeholder", self.strategy_facts())
                        self.assertEqual(failure.exception.code, codes[-1])
                    self.assertEqual(transport.call_count, expected_calls)


if __name__ == "__main__":
    unittest.main()
