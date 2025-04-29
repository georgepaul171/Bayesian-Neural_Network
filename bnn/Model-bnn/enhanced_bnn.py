import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

# Create output directory if it doesn't exist
output_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(output_dir, exist_ok=True)

def save_plot(fig, filename):
    """Helper function to save plots"""
    plt.savefig(os.path.join(output_dir, filename))
    plt.close(fig)

def log_cosh_loss(y_pred, y_true):
    return torch.mean(torch.log(torch.cosh(y_pred - y_true + 1e-6)))

class Enhanced_MC_Dropout_Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.act = nn.LeakyReLU(negative_slope=0.05)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(256, 128)
        self.skip_proj = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(p=0.2)

        self.fc3 = nn.Linear(128, 64)
        self.skip_proj2 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(p=0.2)

        self.out = nn.Linear(64, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.bn1(x)
        x = self.act(self.fc1(x))
        x = self.bn2(x)
        x = self.dropout1(x)

        residual1 = self.skip_proj(x)
        x = self.act(self.fc2(x))
        x = self.bn3(x)
        x = self.dropout2(x)
        x = x + residual1

        residual2 = self.skip_proj2(x)
        x = self.act(self.fc3(x))
        x = self.bn4(x)
        x = self.dropout3(x)
        x = x + residual2

        return self.out(x)
    
def train_model(model, train_loader, val_loader=None, epochs=50, patience=10, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    loss_fn = log_cosh_loss    

    best_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = loss_fn(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    output = model(X_batch)
                    loss = loss_fn(output, y_batch)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth')))
                    break
                    
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}" + 
              (f", Val Loss: {avg_val_loss:.4f}" if val_loader is not None else ""))
        
        scheduler.step(avg_val_loss)
    
    return train_losses, val_losses

def predict_mc(model, X, n_samples=100):
    model.train()  # Keep dropout active
    preds = torch.stack([model(X) for _ in range(n_samples)])
    
    # Epistemic uncertainty (model uncertainty)
    mean = preds.mean(0)
    std = preds.std(0)
    
    # Aleatoric uncertainty (data uncertainty)
    aleatoric_uncertainty = torch.mean((preds - mean) ** 2, dim=0)
    
    return mean.detach().numpy(), std.detach().numpy(), aleatoric_uncertainty.detach().numpy()

def plot_uncertainty_decomposition(mean_preds, epistemic_uncertainty, aleatoric_uncertainty, true_values=None):
    total_uncertainty = np.sqrt(epistemic_uncertainty**2 + aleatoric_uncertainty)
    
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: Total uncertainty vs predictions
    plt.subplot(131)
    plt.scatter(mean_preds, total_uncertainty, alpha=0.5)
    plt.xlabel('Predicted Emissions')
    plt.ylabel('Total Uncertainty')
    plt.title('Total Uncertainty vs Predictions')
    
    # Plot 2: Epistemic vs Aleatoric uncertainty
    plt.subplot(132)
    plt.scatter(epistemic_uncertainty, aleatoric_uncertainty, alpha=0.5)
    plt.xlabel('Epistemic Uncertainty')
    plt.ylabel('Aleatoric Uncertainty')
    plt.title('Uncertainty Decomposition')
    
    # Plot 3: Prediction error vs uncertainty
    if true_values is not None:
        plt.subplot(133)
        errors = np.abs(mean_preds - true_values)
        plt.scatter(total_uncertainty, errors, alpha=0.5)
        plt.xlabel('Total Uncertainty')
        plt.ylabel('Absolute Error')
        plt.title('Uncertainty vs Error')
    
    plt.tight_layout()
    save_plot(fig, 'uncertainty_decomposition.png')

def analyze_risky_suppliers(mean_preds, total_uncertainty, supplier_ids, threshold_emissions=None, threshold_uncertainty=None):
    if threshold_emissions is None:
        threshold_emissions = np.percentile(mean_preds, 90)
    if threshold_uncertainty is None:
        threshold_uncertainty = np.percentile(total_uncertainty, 90)
    
    # Flatten arrays to ensure 1D indexing
    mean_preds = mean_preds.flatten()
    total_uncertainty = total_uncertainty.flatten()
    
    risky_mask = (mean_preds > threshold_emissions) & (total_uncertainty > threshold_uncertainty)
    risky_suppliers = supplier_ids[risky_mask]
    
    print(f"\nIdentified {len(risky_suppliers)} potentially risky suppliers:")
    for idx, supplier_id in enumerate(risky_suppliers):
        print(f"Supplier {supplier_id}:")
        print(f"  Predicted Emissions: {mean_preds[risky_mask][idx]:.2f} tCO2e")
        print(f"  Total Uncertainty: {total_uncertainty[risky_mask][idx]:.2f}")
    
    return risky_suppliers

def feature_importance_analysis(model, X, feature_names):
    model.eval()
    X_tensor = torch.FloatTensor(X)
    X_tensor.requires_grad = True
    
    # Get predictions
    preds = model(X_tensor)
    preds.mean().backward()
    
    # Calculate importance scores
    importance_scores = torch.abs(X_tensor.grad).mean(0).detach().numpy()
    
    # Plot feature importance
    fig = plt.figure(figsize=(10, 6))
    sns.barplot(x=importance_scores, y=feature_names)
    plt.title('Feature Importance Scores')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    save_plot(fig, 'feature_importance.png')
    
    return importance_scores

def plot_predictions_with_error_bars(mean_preds, total_uncertainty, true_values, supplier_ids, top_n=20):
    """
    Plot predictions with error bars for the top N suppliers with highest uncertainty
    """
    # Sort by uncertainty and get top N
    sorted_indices = np.argsort(total_uncertainty.flatten())[-top_n:]
    sorted_means = mean_preds[sorted_indices]
    sorted_uncertainty = total_uncertainty[sorted_indices]
    sorted_true = true_values[sorted_indices]
    sorted_ids = supplier_ids[sorted_indices]
    
    fig = plt.figure(figsize=(12, 6))
    x = np.arange(len(sorted_means))
    
    # Plot predictions with error bars
    plt.errorbar(x, sorted_means.flatten(), yerr=sorted_uncertainty.flatten(), 
                fmt='o', capsize=5, label='Predicted ± Uncertainty')
    plt.scatter(x, sorted_true.flatten(), color='red', label='True Values')
    
    plt.xticks(x, sorted_ids, rotation=45, ha='right')
    plt.xlabel('Supplier ID')
    plt.ylabel('Emissions (tCO2e)')
    plt.title(f'Top {top_n} Suppliers by Uncertainty')
    plt.legend()
    plt.tight_layout()
    save_plot(fig, 'predictions_with_error_bars.png')

def plot_trace_diagnostics(train_losses, val_losses, mean_preds, true_values):
    """
    Plot diagnostic plots including training curves and prediction distributions
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training and validation loss curves
    axes[0, 0].plot(train_losses, label='Training Loss')
    if val_losses:
        axes[0, 0].plot(val_losses, label='Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    
    # Prediction vs True values scatter
    axes[0, 1].scatter(true_values, mean_preds, alpha=0.5)
    min_val = min(true_values.min(), mean_preds.min())
    max_val = max(true_values.max(), mean_preds.max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    axes[0, 1].set_xlabel('True Values')
    axes[0, 1].set_ylabel('Predicted Values')
    axes[0, 1].set_title('Prediction vs True Values')
    axes[0, 1].legend()
    
    # Residual distribution
    residuals = mean_preds - true_values
    axes[1, 0].hist(residuals, bins=30, alpha=0.7)
    axes[1, 0].axvline(x=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('Residuals (Predicted - True)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Residual Distribution')
    
    # QQ plot of residuals
    from scipy import stats
    stats.probplot(residuals.flatten(), dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot of Residuals')
    
    plt.tight_layout()
    save_plot(fig, 'trace_diagnostics.png')

if __name__ == "__main__":
    # Load synthetic data from CSV
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    # Load data
    data_path = "/Users/georgepaul/Desktop/bayesian-proposal/synthetic-data-apple/static-data/apple_supplier_emissions_static.csv"
    df = pd.read_csv(data_path)
    
    # Prepare features and target
    # Convert categorical variables to numerical
    le = LabelEncoder()
    df['Country'] = le.fit_transform(df['Country'])
    df['Sector'] = le.fit_transform(df['Sector'])
    df['Emissions_Trend'] = le.fit_transform(df['Emissions_Trend'])
    df['Risk_Category'] = le.fit_transform(df['Risk_Category'])
    df['High_Risk_Operations'] = df['High_Risk_Operations'].map({'Yes': 1, 'No': 0})
    
    # Select features and target
    feature_columns = ['Country', 'Sector', 'Renewable_Energy_%', 'Data_Confidence', 
                      'Last_Audit_Year', 'High_Risk_Operations', 'Recycled_Materials_%',
                      'Emissions_Trend', 'Traceability_Score', 'Risk_Category']
    
    X = df[feature_columns].values
    y = (df['Emissions_Scope1_tCO2e'] + df['Emissions_Scope2_tCO2e'] + df['Emissions_Scope3_tCO2e']).values.reshape(-1, 1)
    
    supplier_ids = df['Supplier_ID'].values

    # Train/val/test split
    X_train, X_temp, y_train, y_temp, ids_train, ids_temp = train_test_split(X, y, supplier_ids, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test, ids_val, ids_test = train_test_split(X_temp, y_temp, ids_temp, test_size=0.5, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Scale the target variable (emissions)
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_val = y_scaler.transform(y_val)
    y_test = y_scaler.transform(y_test)

    # Convert to torch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    # DataLoaders
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=32)

    # Model
    model = Enhanced_MC_Dropout_Net(input_dim=X_train.shape[1])

    # Train
    print("Training model...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=60, patience=10)

    # Predict with MC Dropout
    print("Predicting with MC Dropout...")
    mean_preds, epistemic_uncertainty, aleatoric_uncertainty = predict_mc(model, X_test_tensor, n_samples=300)

    # After predictions, inverse transform the predictions and true values for plotting
    mean_preds = y_scaler.inverse_transform(mean_preds)
    epistemic_uncertainty = y_scaler.scale_ * epistemic_uncertainty  # Scale uncertainty by the same factor
    aleatoric_uncertainty = y_scaler.scale_ * aleatoric_uncertainty
    total_uncertainty = np.sqrt(epistemic_uncertainty**2 + aleatoric_uncertainty)
    y_test = y_scaler.inverse_transform(y_test)

    # Plot uncertainty decomposition
    plot_uncertainty_decomposition(mean_preds, epistemic_uncertainty, aleatoric_uncertainty, y_test)

    # Analyze risky suppliers
    analyze_risky_suppliers(mean_preds, total_uncertainty, ids_test)

    # Feature importance
    feature_importance_analysis(model, X_train, feature_columns)

    # Plot predictions with error bars
    print("Plotting predictions with error bars...")
    plot_predictions_with_error_bars(mean_preds, total_uncertainty, y_test, ids_test, top_n=20)

    # Plot trace diagnostics
    print("Plotting trace diagnostics...")
    plot_trace_diagnostics(train_losses, val_losses, mean_preds, y_test) 
