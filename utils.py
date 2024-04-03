import os
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import scipy.stats as stats
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pandas.errors import EmptyDataError

from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches


def create_design_matrix(df_train, df_test, features, output_feature):
    X_train, y_train = df_train[features].to_numpy(), df_train[output_feature].to_numpy()
    X_test, y_test = df_test[features].to_numpy(), df_test[output_feature].to_numpy()

    # Scale input data to facilitate training
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def plot_means_variances(y_true, y_means, y_stddevs, save_path=None):
    plt.rc('font', size=14)
    min_vals = np.min([np.min(y_true), np.min(y_means)])
    max_vals = np.max([np.max(y_true), np.max(y_means)])

    plt.figure(figsize=(16, 6))

    # Plot predicted vs true
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_means, alpha = .7, color="0.3", linewidth = 0, s = 2)
    plt.plot([min_vals, max_vals], [min_vals, max_vals], 'k--', color='red')  # Add diagonal line
    plt.title('Fig (a): Predicted vs True Values')
    plt.xlabel('True Power Output')
    plt.ylabel('Predicted Power Output')
    
    def plot_binned_residuals(y_true, residuals, num_bins=20):
        bins = np.linspace(min(y_true), max(y_true), num_bins + 1)

        bin_means = [0]*num_bins
        bin_stddevs = [0]*num_bins

        for i in range(num_bins):
            mask = (y_true >= bins[i]) & (y_true < bins[i + 1])
            if np.any(mask):
                bin_means[i] = np.mean(y_true[mask])
                bin_stddevs[i] = np.sqrt(mean_squared_error(y_means[mask], y_true[mask]))
        return bin_means, bin_stddevs

    bin_means, bin_stddevs = plot_binned_residuals(y_true, y_means, num_bins=20)
    
    # Plot residuals vs true
    plt.subplot(1, 2, 2)
    plt.scatter(y_true, y_stddevs, alpha = .7, color="0.3", linewidth = 0, s = 2, label='Predicted Standard Deviation', zorder=1)
    plt.scatter(bin_means, bin_stddevs, alpha=1, s=50, color='red', label='True Binned Root Mean Squared Error', zorder=2)
    plt.title('Fig (b): Predicted Standard Deviation vs True RMSE')
    plt.xlabel('True Power Output')
    plt.ylabel('Predicted Standard Deviation')
    plt.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.show()

    
def evaluate_and_save_metrics(model_name, y_train, y_test, y_train_pred, y_test_pred, y_train_stddevs=None, y_test_stddevs=None, ci=0.99, output_file="results.csv"):
    z_value = stats.norm.ppf((1 + ci) / 2)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    train_mae = mean_absolute_error(y_train, y_train_pred)    
    test_mae = mean_absolute_error(y_test, y_test_pred) 

    train_percentage_within_interval = "-"
    test_percentage_within_interval = "-"

    if y_train_stddevs is not None and y_test_stddevs is not None:      
        train_lower_bound = y_train_pred - z_value * y_train_stddevs
        train_upper_bound = y_train_pred + z_value * y_train_stddevs

        test_lower_bound = y_test_pred - z_value * y_test_stddevs
        test_upper_bound = y_test_pred + z_value * y_test_stddevs

        train_within_interval = np.sum(np.logical_and(y_train.ravel() >= train_lower_bound, y_train.ravel() <= train_upper_bound))
        test_within_interval = np.sum(np.logical_and(y_test.ravel() >= test_lower_bound, y_test.ravel() <= test_upper_bound))

        train_percentage_within_interval = (train_within_interval / len(y_train.ravel())) * 100
        test_percentage_within_interval = (test_within_interval / len(y_test.ravel())) * 100

    print(f"Train RMSE: {train_rmse:.3f}")
    print(f"Test RMSE: {test_rmse:.3f}")
    print(f"Train MAE: {train_mae:.3f}")
    print(f"Test MAE: {test_mae:.3f}")

    print(f"Percentage of Test Data Points within {ci*100:.2f}% CI: " +
          f"{train_percentage_within_interval}%" if isinstance(train_percentage_within_interval, str) else f"{train_percentage_within_interval:.2f}%")
    print(f"Percentage of Test Data Points within {ci*100:.2f}% CI: " +
          f"{test_percentage_within_interval}%" if isinstance(test_percentage_within_interval, str) else f"{test_percentage_within_interval:.2f}%")
    
    if model_name is not None:
        new_row = pd.DataFrame({
            "Model Name": [model_name],
            "Train RMSE": [round(train_rmse, 2)],
            "Train MAE": [round(train_mae, 2)],
            "Test RMSE": [round(test_rmse, 2)],
            "Test MAE": [round(test_mae, 2)],
            f"Test % within {ci*100:.2f}% CI": [test_percentage_within_interval if isinstance(test_percentage_within_interval, str) \
                                                else round(test_percentage_within_interval, 2)]
        })

    try:
        results_df = pd.read_csv(output_file)
    except FileNotFoundError or EmptyDataError:
        results_df = pd.DataFrame(columns=list(new_row.columns))
    if model_name in results_df["Model Name"].values:
        results_df.loc[results_df["Model Name"] == model_name] = new_row.values
    elif model_name is not None:
        results_df = pd.concat([results_df, new_row], ignore_index=True)
    results_df.to_csv(output_file, index=False)

    
def add_coverage_prob_plot(plot, label, y_test_pred, y_test_std, y_test, bins=20):    
    # Compute the t-values of the confidence intervals based on Z-scores
    t_values = np.array([stats.norm.ppf(i/bins + (1-i/bins)/2) for i in range(1, bins+1)])

    percentages_within_interval = []
    for t_value in t_values:
        lower_bounds = y_test_pred.ravel() - t_value * y_test_std
        upper_bounds = y_test_pred.ravel() + t_value * y_test_std

        # Count number of data points within the confidence interval
        is_within_interval = np.logical_and(y_test >= lower_bounds, y_test <= upper_bounds)
        num_within_interval = np.sum(is_within_interval)

        # Calculate the percentage of data points within the confidence interval
        percentage_within_interval = (num_within_interval / len(y_test)) * 100
        percentages_within_interval.append(percentage_within_interval)

    plot.scatter(np.arange(1, bins+1)*100/bins, percentages_within_interval, label=f'Percentage: {label}')
    
    
def plot_confidence_interval_scatter(y_test_pred, y_test_std, y_test, bins=20, save_path=None):
    plt.rc('font', size=14)
    
    # Compute the t-values of the confidence intervals based on Z-scores
    t_values = np.array([stats.norm.ppf(i/bins + (1-i/bins)/2) for i in range(1, bins+1)])

    percentages_within_interval = []
    for t_value in t_values:
        lower_bounds = y_test_pred.ravel() - t_value * y_test_std
        upper_bounds = y_test_pred.ravel() + t_value * y_test_std

        # Count number of data points within the confidence interval
        is_within_interval = np.logical_and(y_test >= lower_bounds, y_test <= upper_bounds)
        num_within_interval = np.sum(is_within_interval)

        # Calculate the percentage of data points within the confidence interval
        percentage_within_interval = (num_within_interval / len(y_test)) * 100
        percentages_within_interval.append(percentage_within_interval)

    plt.figure(figsize=(8, 8))
    plt.scatter(np.arange(1, bins+1)*100/bins, percentages_within_interval, color='blue', label='Percentage of Residuals within Interval')
    
    # Plot the expected diagonal line (red line)
    plt.plot([0, 100], [0, 100], color='red', linestyle='--', label='Expected')

    # Add percentage symbols to x-axis ticks
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}%'))

    plt.xlabel('Confidence Intervals')
    plt.ylabel('Percentage within Interval')
    plt.title('Scatter Plot of Percentage of Residuals within the Confidence Intervals')
    plt.legend()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.show()
   
    
def load_dataset_train_test_split(df, features, output_feature):
    keras.utils.set_random_seed(812)
    X = df[features]
    y = df[output_feature]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

    # Scale input data to facilitate training
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, np.array(y_train), np.array(y_test), scaler
   
    
def train_model(model, X_train, y_train, patience, epochs, batch_size, cp_callback, seed):
    tf.random.set_seed(seed)
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    model.summary()
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[early_stopping, cp_callback])
    return history


def train_multivariate_model(model, X_train, y_train, epochs, batch_size, patience, cp_callback):
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    model.build(X_train.shape)

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stopping, cp_callback]
    )

    return history


def plot_loss_history(history):
    plt.plot(history.history['loss'][1:], label='Training Loss')
    plt.plot(history.history['val_loss'][1:], label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
    
    
def compute_predictions(model, X_train, X_test, num_samples=100):
    y_train_pred = []
    y_test_pred = []
    for _ in range(num_samples):
        y_train_pred.append(model.predict(X_train))
        y_test_pred.append(model.predict(X_test))
        
    y_train_pred = np.concatenate(y_train_pred, axis=1)
    y_test_pred = np.concatenate(y_test_pred, axis=1)

    y_train_pred_mean = np.mean(y_train_pred, axis=1)
    y_train_pred_stddevs = np.std(y_train_pred, axis=1)
    
    y_test_pred_mean = np.mean(y_test_pred, axis=1)
    y_test_pred_stddevs = np.std(y_test_pred, axis=1)
    
    return y_train_pred_mean, y_train_pred_stddevs, y_test_pred_mean, y_test_pred_stddevs

def NLL(y, distr): 
    return -distr.log_prob(y) 


# We add 0.001 to the standard deviation to ensure it does not converge to 0 and destabilizes training because the gradient
# of maximum likelihood estimation requires the inversion of the variance. We also activate the parameters using a softplus
# activation function to enfore a positive standard deviation estimate.
def normal_softplus(params): 
    return tfd.Normal(loc=params[:, 0:1], scale=1e-3 + tf.math.softplus(0.05 * params[:, 1:2]))


def multivariate_covariance_normal_softplus(mean_params, std_params, d): 
    means = mean_params
    stds = 1e-3 + tf.math.softplus(0.05 * std_params)
    
    return tfd.MultivariateNormalTriL(loc=means, scale_tril=tfp.math.fill_triangular(stds))


def multivariate_diagonal_normal_softplus(mean_params, std_params, d): 
    means = mean_params
    stds = 1e-3 + tf.math.softplus(0.05 * std_params)
    
    return tfd.MultivariateNormalDiag(loc=means, scale_diag=stds)


def train_test_split_by_turbine(group, test_size=0.2):
    train_set, test_set = train_test_split(group, test_size=test_size, random_state=42)
    return train_set, test_set


def plot_power_over_all_features(df, units, features, output_feature, sample_size=5000):
    df_sampled = df.sample(min(sample_size, len(df)))
    
    num_cols = 5
    num_rows = math.ceil(len(features) / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 20))
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        axes[i].scatter(x=df_sampled[feature], y=df_sampled[output_feature], alpha=0.7, color="0.3", linewidth=0, s=2)
        axes[i].set_title(f'Power/\n{feature}')
        axes[i].set_xlabel(units[feature])
        axes[i].set_ylabel('kW')

    plt.tight_layout()
    plt.show()


def overwrite(model_filepath):    
    if os.path.exists(model_filepath):
        os.remove(model_filepath)

    return model_filepath


def plot_confidence_interval_bar(y_test_pred, y_test_std, y_test, bins=20, save_path=None):
    plt.rc('font', size=14)
    
    # Compute the t-values of the confidence intervals based on Z-scores
    t_values = np.array([stats.norm.ppf(i/bins + (1-i/bins)/2) for i in range(1, bins+1)])

    percentages_within_interval = []
    for t_value in t_values:
        lower_bounds = y_test_pred.ravel() - t_value * y_test_std
        upper_bounds = y_test_pred.ravel() + t_value * y_test_std

        # Count number of data points within the confidence interval
        is_within_interval = np.logical_and(y_test >= lower_bounds, y_test <= upper_bounds)
        num_within_interval = np.sum(is_within_interval)

        # Calculate the percentage of data points within the confidence interval
        percentage_within_interval = (num_within_interval / len(y_test)) * 100
        percentages_within_interval.append(percentage_within_interval)

    plt.figure(figsize=(8, 8))
    # Plotting histogram
    plt.bar(np.arange(1, bins+1)*100/bins, percentages_within_interval, color='#76b5c5', width=80/bins, edgecolor='black', alpha=0.9, label='Percentage of Residuals within Interval')
    
    # Plot the expected diagonal line (red line)
    plt.plot([0, 100], [0, 100], color='red', linestyle='--', label='Expected')
    
    # Calculate differences between the blue bars and the expected line
    expectations = np.arange(1, bins+1)*100/bins
    differences = np.array(percentages_within_interval) - expectations

    # Plot individual red bars for each discrepancy
    for i, difference in enumerate(differences):
        if difference != 0:
            plt.bar((i+1)*100/bins, abs(difference), bottom=min(percentages_within_interval[i], expectations[i]), color='red', width=80/bins, edgecolor='black', alpha=0.3)

    handles, _ = plt.gca().get_legend_handles_labels()

    plt.xlabel('Confidence Intervals')
    plt.ylabel('Percentage within Interval (%)')
    plt.title('Histogram of Percentage of Residuals within the Confidence Intervals')
    # plt.legend()
    red_patch = mpatches.Patch(color='red', alpha=0.3, label=f'Gap (MCE={max(abs(differences)):.2f})')
    handles.append(red_patch)
    plt.legend(handles=handles)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.show()


def save_preds(name, y_test_pred, y_test_stddevs, filename="preds.csv"):
    data = {
        'Model Name': [name] * len(y_test_pred),
        'y_test_pred': y_test_pred,
        'y_test_stddevs': y_test_stddevs
    }
    df_new = pd.DataFrame(data)
    
    try:
        df_existing = pd.read_csv(filename)
        if name in df_existing['Model Name'].values:
            # If exists, overwrite it
            df_existing = df_existing[df_existing['Model Name'] != name]
            df_existing = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            # If not exists, append the new data
            df_existing = pd.concat([df_existing, df_new], ignore_index=True)
    except FileNotFoundError or EmptyDataError:
        # If file doesn't exist, create new dataframe
        df_existing = df_new

    # Save to file
    df_existing.to_csv(filename, index=False)
    return df_existing


