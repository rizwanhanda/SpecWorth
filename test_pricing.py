"""Offline regression tests for the shipped bundle and explanations."""
import unittest
from unittest.mock import patch

from matplotlib.figure import Figure

from app import briefing, configure, main, show_optional_takeaway, st
from explanations import draw_step, feature_values, ranked_features, waterfall_png
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
                side_effect=["This laptop", "Across laptops"]), patch.object(st, "subheader") as heading, \
                patch.object(st, "button") as button:
            main()
            main()
            self.assertEqual(sum(call.args[0] == "Buyer’s briefing" for call in heading.call_args_list), 2)
            button.assert_not_called()

    def test_briefing_matches_the_waterfall_without_gemini(self):
        row = dict(self.row, Weight=2.05)
        price, baseline, _, _, detail = explain(self.model, self.catalog, row)
        with patch.object(st, "secrets", {}), patch.object(st, "markdown") as markdown, \
                patch.object(st, "write") as write:
            briefing(row, price, baseline, ranked_features(detail), feature_values(row, detail), self.catalog)
        content = "\n".join(call.args[0] for call in markdown.call_args_list)
        self.assertIn("€1,027.49 estimated catalog price", content)
        self.assertIn("€80.53 below the €1,108.02 model baseline", content)
        self.assertIn("Solid-state storage · 256 GB · **+€48.22**", content)
        self.assertIn("IPS panel · Yes · **+€33.86**", content)
        self.assertIn("Weight · 2.05 kg · **−€72.28**", content)
        self.assertIn("Form factor · Notebook · **−€62.42**", content)
        self.assertEqual(sum(line.startswith("- ") for line in content.splitlines()), 4)
        self.assertIn("€191 on average", write.call_args.args[0])
        self.assertIn("258 test laptops", write.call_args.args[0])

    def test_gemini_tip_stays_tied_to_configuration(self):
        with patch.object(st, "secrets", {"GEMINI_API_KEY": "placeholder"}), \
             patch.object(st, "session_state", {}), patch.object(st, "button", side_effect=[True, False, False]), \
             patch("app.generate_takeaway", return_value="Sample") as generate, patch.object(st, "text") as text:
            for ram in (8, 8, 16):
                show_optional_takeaway({"Ram": ram}, {"Memory": f"{ram} GB"})
            generate.assert_called_once_with("placeholder", {"Memory": "8 GB"})
            self.assertEqual(text.call_count, 2)


if __name__ == "__main__":
    unittest.main()
