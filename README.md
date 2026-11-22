# SpecWorth

Configure a laptop to estimate its catalog price in euros and explore the prediction with TreeSHAP. The default Dell notebook lands at €1,037.73, starting from a model baseline of €1,108.02.

![SHAP waterfall for the default Dell notebook, with signed euro contributions and the final estimate](shap-preview.png)

## Explore the explanation

- **This laptop** shows the seven largest feature contributions. The remaining contributions share one row. Select **Show every feature** to expand the chart.
- **Inspect a feature** shows its selected value, signed contribution, and the middle 80% of contributions observed on held-out laptops. Open the underlying inputs to see which category indicators were active or absent.
- **Across laptops** ranks features by their mean absolute contribution over the 258 held-out configurations. Use the sidebar to compare full predictions when evaluating a hardware change.
- **Buyer’s briefing** uses the Google Gemini API when enabled. Clicking Generate sends the selected specs, estimated price, baseline, and component totals to Google.

## Model performance

The XGBoost regressor was fitted on 1,045 of the catalog’s 1,303 rows. Configurations with identical broad hardware specs were kept together. Candidate selection used two repetitions of grouped five-fold cross-validation on the development set. CPU and GPU line features let the trees share information between related components.

| Evaluation | Mean absolute error |
| --- | ---: |
| Repeated grouped cross-validation | €158.23 |
| Separate five-fold check holding out named products within the development set | €180.84 |
| 258-row configuration holdout | €191.38 |

Holdout R² was 0.808. The 90th percentile absolute error was €462.41. The same holdout has been used across project revisions. Prices and performance reflect this catalog. Validation against later retail listings would be needed for current-market estimates.

## Run locally

Use Python 3.14 in a virtual environment.

```shell
python -m pip install -r requirements.txt
python -m unittest -v test_pricing
python -m streamlit run app.py
```

For a Gemini briefing, set `GEMINI_API_KEY` in your local `.streamlit/secrets.toml`. The estimator and SHAP explorer work with the model archive alone and run locally.

## Deploy to Streamlit Community Cloud

Select `app.py` as the entry point, choose Python 3.14 under Advanced settings, and enter `GEMINI_API_KEY` in the app’s Secrets settings if you want briefings. Keep the local secrets file private. The published project serves the model in `laptop_model.zip`. The training script and source catalog are kept in the original local workspace.
