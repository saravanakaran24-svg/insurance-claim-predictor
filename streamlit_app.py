import joblib, numpy as np, pandas as pd, streamlit as st

st.set_page_config(page_title='Vehicle Insurance Claim Predictor')
st.title('Vehicle Insurance Claim Predictor')

@st.cache_resource
def load_model():
    return joblib.load('claim_model.joblib')

model = load_model()
REF = pd.Timestamp('2021-05-22')
COMPANIES = ['A', 'AA', 'AC', 'B', 'BB', 'BC', 'BQ', 'C', 'DA', 'O', 'RE']

company = st.selectbox('Insurance company', COMPANIES)
cost = st.number_input('Cost of vehicle', value=40000.0)
min_cov = st.number_input('Min coverage', value=900.0)
max_cov = st.number_input('Max coverage', value=10000.0)
expiry = st.date_input('Expiry date', value=pd.Timestamp('2026-12-31'))

if st.button('Predict'):
    d = pd.Timestamp(expiry)
    row = pd.DataFrame([{
        'Cost_of_vehicle': cost, 'Min_coverage': min_cov, 'Max_coverage': max_cov,
        'Days_to_expiry': (d - REF).days, 'Expiry_year': d.year, 'Expiry_month': d.month,
        'Coverage_ratio': max_cov / min_cov if min_cov else np.nan,
        'Coverage_to_cost': max_cov / cost if cost else np.nan,
        'Specs_missing': 0, 'Insurance_company': company}])
    p = float(model.predict_proba(row)[0, 1])
    st.metric('Claim probability', f'{p:.1%}')
    st.write('Prediction:', '**Claim (1)**' if p >= 0.5 else '**No claim (0)**')
