# Predictive Forecasting of Care Load & Placement Demand

## Files
- `UAC_Forecasting_Colab.ipynb` — complete Google Colab training workflow
- `app.py` — Streamlit dashboard
- `requirements.txt` — Python dependencies
- `artifacts/` — trained models and cleaned data

## How to use
1. Open the notebook in Google Colab.
2. Upload `HHS_Unaccompanied_Alien_Children_Program.csv`.
3. Run all cells.
4. Download `uac_streamlit_artifacts.zip`.
5. Extract the `artifacts` folder into the same folder as `app.py`.
6. Run:
   `pip install -r requirements.txt`
7. Start:
   `streamlit run app.py`

## GitHub + Streamlit deployment
Upload:
- `app.py`
- `requirements.txt`
- `artifacts/daily_data.csv`
- `artifacts/models.pkl`
- `artifacts/validation_residuals.pkl`
- `artifacts/model_comparison.csv`

Then create a Streamlit Community Cloud app using `app.py`.

The dashboard is intentionally kept simple and student-project oriented rather than looking like a commercial product.
