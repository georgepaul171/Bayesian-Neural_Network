import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel, DotProduct, ConstantKernel as C
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error

#  Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "generated-data/synthetic_supplier_emissions.csv")
preprocessor_path = os.path.join(base_dir, "models/preprocessor.joblib")
target_scaler_path = os.path.join(base_dir, "models/target_scaler.joblib")

# Load Data 
df = pd.read_csv(data_path)
target_col = 'Reported_Emissions_kgCO2e'
cat_cols = ['Country', 'Product_Type', 'Material_Type', 'Source', 'Data_Confidence', 'Report_Month']
num_cols = ['Units_Produced', 'Emission_Factor', 'ThirdParty_Estimate', 'Green_Certified', 'Prior_Violations', 'Audit_Score']

# Load Preprocessors
preprocessor = joblib.load(preprocessor_path)
target_scaler = joblib.load(target_scaler_path)

# Preprocess Data
X_full = preprocessor.transform(df[cat_cols + num_cols]).toarray()
y_full = target_scaler.transform(df[[target_col]]).flatten()

# PCA for Dimensionality Reduction 
pca = PCA(n_components=0.99)  # retain 99% variance
X_pca = pca.fit_transform(X_full)

# Train/Test Split 
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_full, test_size=0.2, random_state=42)

# Enhanced GPR Kernel
kernel = C(1.0, (1e-3, 1e3)) * (
    Matern(length_scale=1.0, nu=1.5) + 
    RBF(length_scale=1.0) +
    DotProduct() +
    WhiteKernel(noise_level=1)
)

# GPR Model
gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, n_restarts_optimizer=15, normalize_y=True)

print("Training Tuned GPR...")
gpr.fit(X_train, y_train)

print(f"Optimized Kernel: {gpr.kernel_}")

# Predict
print("Predicting on test data...")
y_pred, y_std = gpr.predict(X_test, return_std=True)

# Inverse transform to original units 
y_pred_orig = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_std_orig = y_std * target_scaler.scale_[0]
y_true_orig = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

#  Plot 1: Predicted Emissions vs Uncertainty 
plt.figure(figsize=(8, 5))
plt.scatter(y_pred_orig, y_std_orig, alpha=0.6)
plt.xlabel("Predicted Emissions (kgCO2e)")
plt.ylabel("Uncertainty (± kgCO2e)")
plt.title("GPR (Tuned): Emission vs Uncertainty")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "tuned_emission_vs_uncertainty.png"))
plt.show()

#  Plot 2: Uncertainty Distribution
plt.figure(figsize=(7, 4))
plt.hist(y_std_orig, bins=30, edgecolor='black', color='orange')
plt.title("Distribution of Predictive Uncertainty (Tuned GPR)")
plt.xlabel("Uncertainty (kgCO₂e)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "tuned_uncertainty_histogram.png"))
plt.show()

# Evaluation 
mae = mean_absolute_error(y_true_orig, y_pred_orig)
rmse = mean_squared_error(y_true_orig, y_pred_orig) ** 0.5

print(f"\nMAE: {mae:,.2f} kgCO₂e")
print(f"RMSE: {rmse:,.2f} kgCO₂e")

from scipy.stats import norm

def prob_emission_above(gpr_model, x_new, threshold, scaler, pca, samples=10000):
    """
    Estimate P(emissions > threshold) for a new input using GPR predictive mean and std.
    
    Parameters:
    - gpr_model: trained GaussianProcessRegressor
    - x_new: DataFrame with one row (same features as training)
    - threshold: threshold in original (kgCO2e) units
    - scaler: target scaler used during training
    - pca: PCA model used to transform input features
    - samples: number of draws for Monte Carlo estimation (default: 10k)
    
    Returns:
    - Probability (float) that predicted emissions > threshold
    """
    # Transform new input
    x_transformed = preprocessor.transform(x_new).toarray()
    x_pca = pca.transform(x_transformed)

    # Predict mean and std (in scaled space)
    mu, std = gpr_model.predict(x_pca, return_std=True)
    
    # Monte Carlo sampling
    y_samples_scaled = np.random.normal(loc=mu[0], scale=std[0], size=samples)
    y_samples = scaler.inverse_transform(y_samples_scaled.reshape(-1, 1)).flatten()

    # Compute probability
    prob = (y_samples > threshold).mean()
    return prob

#  Probability Query Example 
new_supplier = df.iloc[[0]][cat_cols + num_cols]
threshold_val = 100000

prob = prob_emission_above(
    gpr_model=gpr,
    x_new=new_supplier,
    threshold=threshold_val,
    scaler=target_scaler,
    pca=pca
)

print(f"\nP(emissions > {threshold_val:,} kgCO₂e) for Supplier 0: {prob:.2%}")


## python gpr-emissions.py


# Training Tuned GPR...
# Optimized Kernel: 0.202**2 * Matern(length_scale=2.81, nu=1.5) + RBF(length_scale=2.44) + DotProduct(sigma_0=0.00243) + WhiteKernel(noise_level=1.87)
# Predicting on test data...
# 2025-04-20 16:44:22.549 python[35887:5518071] +[IMKClient subclass]: chose IMKClient_Modern
# 2025-04-20 16:44:22.549 python[35887:5518071] +[IMKInputSession subclass]: chose IMKInputSession_Modern

# MAE: 15,148.37 kgCO₂e
# RMSE: 24,875.24 kgCO₂e

# P(emissions > 100,000 kgCO₂e) for Supplier 0: 28.89%