import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def predict_prob_below(x_new_df, threshold, trace, preprocessor, target_scaler, show_plot=True):
    """
    Predicts the probability that emissions are below a given threshold
    using a strict Bayesian Neural Network (BNN).

    Parameters:
        x_new_df (pd.DataFrame): 1-row input with supplier features
        threshold (float): The emissions threshold to evaluate
        trace (dict): Posterior samples from PyMC approx.sample(..., return_inferencedata=False)
        preprocessor: Trained sklearn ColumnTransformer
        target_scaler: Trained sklearn StandardScaler for the target variable
        show_plot (bool): Whether to plot the predictive distribution

    Returns:
        dict: {
            'mean': ...,
            'std': ...,
            'probability_below_threshold': ...
        }
    """

    # Step 1: Preprocess
    x_new = preprocessor.transform(x_new_df).toarray()

    # Step 2: Manual forward passes
    y_samples = []
    for i in range(len(trace["w1"])):
        w1_ = trace["w1"][i]
        b1_ = trace["b1"][i]
        w2_ = trace["w2"][i]
        b2_ = trace["b2"][i]
        w3_ = trace["w3"][i]
        b3_ = trace["b3"][i]

        h1 = np.tanh(np.dot(x_new, w1_) + b1_)
        h2 = np.tanh(np.dot(h1, w2_) + b2_)
        pred = np.dot(h2, w3_) + b3_
        y_samples.append(pred.item())

    y_samples = np.array(y_samples)
    y_samples_orig = target_scaler.inverse_transform(y_samples.reshape(-1, 1)).flatten()

    # Step 3: Stats
    mean = y_samples_orig.mean()
    std = y_samples_orig.std()
    prob_below = (y_samples_orig < threshold).mean()

    if show_plot:
        sns.kdeplot(y_samples_orig)
        plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
        plt.title("Predictive Distribution for New Input")
        plt.xlabel("Predicted Emissions (kgCO₂e)")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.show()

    return {
        "mean": mean,
        "std": std,
        "probability_below_threshold": prob_below
    }