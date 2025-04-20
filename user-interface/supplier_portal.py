import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
import joblib
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.nn.functional as F

class MC_Dropout_Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(p=0.2)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.out(x)


preprocessor = joblib.load('/Users/georgepaul/Desktop/bayesian-proposal/models/preprocessor.joblib')
target_scaler = joblib.load('/Users/georgepaul/Desktop/bayesian-proposal/models/target_scaler.joblib')

model_path = '/Users/georgepaul/Desktop/bayesian-proposal/models/mc_dropout_model.pth'
input_dim = len(preprocessor.get_feature_names_out())

model = MC_Dropout_Net(input_dim)
model.load_state_dict(torch.load(model_path))
model.train()  # Keep dropout active for MC sampling

# --- MC Dropout prediction function ---
def predict_mc(model, X, n_samples=50):
    preds = torch.stack([model(X) for _ in range(n_samples)])
    mean = preds.mean(0)
    std = preds.std(0)
    return mean.detach().numpy(), std.detach().numpy()

# --- Streamlit Interface ---
st.title("Supplier Emissions Portal")
st.write("Upload your emissions report to estimate emissions and assess audit risk.")

uploaded = st.file_uploader("Upload your CSV", type="csv")

if uploaded:
    try:
        supplier_df = pd.read_csv(uploaded)

        # Preprocess new supplier data
        X_supplier = preprocessor.transform(supplier_df)
        X_tensor = torch.tensor(X_supplier.toarray(), dtype=torch.float32)

        # Make MC predictions
        mean_pred, uncertainty = predict_mc(model, X_tensor)
        mean_pred_orig = target_scaler.inverse_transform(mean_pred)
        uncertainty_orig = uncertainty * target_scaler.scale_[0]

        # Compose results
        results = pd.DataFrame({
            "Predicted Emissions (kgCO₂e)": mean_pred_orig.flatten(),
            "Uncertainty (± kgCO₂e)": uncertainty_orig.flatten(),
            "Audit Likelihood": uncertainty_orig.flatten() > 20000
        })

        st.subheader("Prediction Results")
        st.dataframe(results)

        # Download option
        st.download_button(
            label="Download CSV",
            data=results.to_csv(index=False).encode(),
            file_name="supplier_emissions_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")