"""Offline regression tests for the shipped bundle and explanations."""
import unittest
from unittest.mock import patch

from app import briefing, configure, main, st
from explanations import feature_values, ranked_features, waterfall_png
from pricing import explain, load_bundle


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

    def test_app_views_without_credentials(self):
        with patch.object(st, "secrets", {}), patch.object(st, "segmented_control",
                side_effect=["This laptop", "Across laptops"]), patch.object(st, "info") as info:
            main()
            main()
            self.assertEqual(info.call_count, 2)

    def test_briefing_stays_tied_to_configuration(self):
        with patch.object(st, "secrets", {"GEMINI_API_KEY": "placeholder"}), \
             patch.object(st, "session_state", {}), patch.object(st, "button", side_effect=[True, False, False]), \
             patch("app.generate_briefing", return_value="Sample") as generate, patch.object(st, "text") as text:
            for ram in (8, 8, 16):
                briefing({"Ram": ram}, 1000, 900, {"Memory": 100})
            generate.assert_called_once()
            self.assertEqual(text.call_count, 2)


if __name__ == "__main__":
    unittest.main()
