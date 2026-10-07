import gradio as gr, joblib, numpy as np, pandas as pd

model = joblib.load('claim_model.joblib')
REF = pd.Timestamp('2021-05-22')
COMPANIES = ['A', 'AA', 'AC', 'B', 'BB', 'BC', 'BQ', 'C', 'DA', 'O', 'RE']

def predict(company, cost, min_cov, max_cov, expiry):
    d = pd.Timestamp(expiry)
    row = pd.DataFrame([{
        'Cost_of_vehicle': cost, 'Min_coverage': min_cov, 'Max_coverage': max_cov,
        'Days_to_expiry': (d - REF).days, 'Expiry_year': d.year, 'Expiry_month': d.month,
        'Coverage_ratio': max_cov / min_cov if min_cov else np.nan,
        'Coverage_to_cost': max_cov / cost if cost else np.nan,
        'Specs_missing': 0, 'Insurance_company': company}])
    p = float(model.predict_proba(row)[0, 1])
    return {'Claim (1)': p, 'No claim (0)': 1 - p}

demo = gr.Interface(
    predict,
    [gr.Dropdown(COMPANIES, value='A', label='Insurance company'),
     gr.Number(value=40000, label='Cost of vehicle'),
     gr.Number(value=900, label='Min coverage'),
     gr.Number(value=10000, label='Max coverage'),
     gr.Textbox(value='2026-12-31', label='Expiry date (YYYY-MM-DD)')],
    gr.Label(),
    title='Vehicle Insurance Claim Predictor')

if __name__ == '__main__':
    demo.launch()
