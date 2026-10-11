# SmartCart Streamlit Dashboard

This dashboard turns `smartcart.ipynb` into an interactive Streamlit app for SmartCart customer segmentation.

## Run

```powershell
cd "C:\Users\Aarav Sharma\Documents\Codex\2026-10-11\i-wa\outputs\smartcart_streamlit_dashboard"
pip install -r requirements.txt
streamlit run app.py
```

## Data

The app expects the dataset used by the notebook: `smartcart_customers.csv`.

You can either:

- upload the CSV from the sidebar after the app starts, or
- place `smartcart_customers.csv` in this folder and refresh the dashboard.

## Included Views

- Data preparation and cleaning summary
- Feature distributions and correlation heatmap
- Elbow and silhouette analysis
- K-Means, Ward clustering, and GMM comparison
- Cluster profile table, box plots, radar chart, and income-vs-spending view
- Product category spend and campaign response validation
- Persona-based marketing action cards
- PCA two-dimensional cluster view
