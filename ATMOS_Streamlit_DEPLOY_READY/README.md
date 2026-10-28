# ATMOS — Weather & Climate Intelligence

A Streamlit dashboard combining live weather lookup and an educational climate regression demonstration.

## Deploy on Streamlit Community Cloud

1. Create a **new public GitHub repository**.
2. Upload the contents of this folder to the **repository root**. The repository root must contain `app.py`, `requirements.txt`, `README.md`, and the `data` folder.
3. In Streamlit Community Cloud, choose that repository, branch `main`, and main file path `app.py`.
4. Deploy. Do not enter `requirements.txt` as a dependency inside `requirements.txt`, and do not use a nested main file path.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Required dataset columns

- `Avg Temperature (°C)`
- `CO2 Emissions (Tons/Capita)`
- `Rainfall (mm)`
- `Sea Level Rise (mm)`

The included dataset is stored at `data/climate_change_dataset.csv`.

## Limitations

The app reports test-set RMSE and R² for baseline Linear Regression models. Its rainfall and sea-level estimates are educational outputs, not validated climate forecasts. Live weather lookup requires internet access.
