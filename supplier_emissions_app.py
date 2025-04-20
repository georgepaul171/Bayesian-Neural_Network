# supplier_emissions_app.py (Streamlit App)

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# Load trained model and preprocessing
class MC_Dropout_Net(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.out = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.out(x)

# Load preprocessing pipeline (assumes pre-fitted and saved)
preprocessor = joblib.load('/Users/georgepaul/Desktop/bayesian-proposal/models/preprocessor.joblib')
target_scaler = joblib.load('/Users/georgepaul/Desktop/bayesian-proposal/models/target_scaler.joblib')

# Sample input columns
cat_cols = ['Country', 'Product_Type', 'Material_Type', 'Source', 'Data_Confidence', 'Report_Month']
num_cols = ['Units_Produced', 'Emission_Factor', 'ThirdParty_Estimate', 'Green_Certified', 'Prior_Violations', 'Audit_Score']
input_dim = preprocessor.transform(pd.DataFrame([{
    'Country': 'USA', 'Product_Type': 'Phone', 'Material_Type': 'Metal',
    'Source': 'Direct', 'Data_Confidence': 'High', 'Report_Month': 'Jan',
    'Units_Produced': 1000, 'Emission_Factor': 0.5, 'ThirdParty_Estimate': 1.0,
    'Green_Certified': 0, 'Prior_Violations': 0, 'Audit_Score': 90
}])).shape[1]

model = MC_Dropout_Net(input_dim)
model.load_state_dict(torch.load('/Users/georgepaul/Desktop/bayesian-proposal/models/mc_dropout_model.pth'))
model.train()

# Inference function with MC Dropout
def predict_with_uncertainty(model, X, n_samples=50):
    preds = torch.stack([model(X) for _ in range(n_samples)])
    mean = preds.mean(0)
    std = preds.std(0)
    return mean.detach().numpy(), std.detach().numpy()

# Streamlit UI
st.title("🌍 Supplier Emissions Risk Estimator")
st.markdown("Estimate emissions, uncertainty, and risk score for supplier profiles.")

user_input = {}
for col in cat_cols:
    user_input[col] = st.selectbox(f"{col}", options=['USA', 'China', 'India', 'Germany'] if col == 'Country' else ['A', 'B', 'C', 'D'])
for col in num_cols:
    user_input[col] = st.number_input(f"{col}", min_value=0.0, value=10.0 if col != 'Audit_Score' else 80.0)

if st.button("Predict Emissions"):
    input_df = pd.DataFrame([user_input])
    X_processed = preprocessor.transform(input_df)
    X_tensor = torch.tensor(X_processed.toarray(), dtype=torch.float32)
    
    mean, std = predict_with_uncertainty(model, X_tensor, n_samples=50)
    mean_orig = target_scaler.inverse_transform(mean)
    std_orig = std * target_scaler.scale_[0]

    risk_score = mean_orig.flatten()[0] * std_orig.flatten()[0]

    st.success(f"**Predicted Emissions**: {mean_orig[0][0]:,.2f} kgCO₂e")
    st.info(f"**Model Uncertainty**: ± {std_orig[0][0]:,.2f} kgCO₂e")
    st.warning(f"**Risk Score**: {risk_score:,.2f}")
    st.caption("Lower uncertainty reflects higher data confidence. Risk score = Emissions × Uncertainty")

# Optionally expand to show SHAP in app later


# To run: streamlit run supplier_emissions_app.py --server.fileWatcherType none