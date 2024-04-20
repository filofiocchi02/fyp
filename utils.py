import os
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import scipy.stats as stats
import numpy as np
import pandas as pd
from joblib import load

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

from IPython.display import display, clear_output
import ipywidgets as widgets

from constants import *


def create_design_matrix(df_train: pd.DataFrame, df_test: pd.DataFrame, features, output_feature):
    """Generates design matrix.

    Args:
        df_train (pd.DataFrame): training dataframe
        df_test (pd.DataFrame): testing dataframe
        features (list): input features
        output_feature (str): output feature

    Returns:
        Tuple: desing matrices and scaler object
    """
    X_train, y_train = df_train[features].to_numpy(), df_train[output_feature].to_numpy()
    X_test, y_test = df_test[features].to_numpy(), df_test[output_feature].to_numpy()

    # Scale input data to facilitate training
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def plot_binned_residuals(y_true, y_means, num_bins=20):
    """Generates bins and their means and standard deviations

    Args:
        y_true (np.array): _description_
        y_means (np.array): _description_
        num_bins (int, optional): number of bins. Defaults to 20.

    Returns:
        _type_: means and standard deviations of bins
    """
    bins = np.linspace(min(y_true), max(y_true), num_bins + 1)

    bin_means = [0]*num_bins
    bin_stddevs = [0]*num_bins

    for i in range(num_bins):
        mask = (y_true >= bins[i]) & (y_true < bins[i + 1])
        if np.any(mask):
            bin_means[i] = np.mean(y_true[mask])
            bin_stddevs[i] = np.sqrt(mean_squared_error(y_means[mask], y_true[mask]))
    return bin_means, bin_stddevs


def plot_means_variances(y_true, y_means, y_stddevs, save_path=None):
    """Plots two plots: predicted meam vs true and predicted std vs true.

    Args:
        y_true (np.array): output true values
        y_means (np.array): output predicted values
        y_stddevs (np.array): output predicted standard deviation
        save_path (str, optional): Where to save the plot. Defaults to None.
    """
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


def calc_percentages_in_interval(y_train, y_test, y_train_pred, y_test_pred, y_train_stddevs, y_test_stddevs, ci):
    """Given a confidence interval it calculates the percentage of true outputs within
    the bounds of the confidence interval of the predicted density.

    Args:
        y_train (np.array): output true values (train set)
        y_test (np.array): output true values (test set)
        y_train_pred (np.array): output mean predictions (train set)
        y_test_pred (np.array): output mean predictions (test set)
        y_train_stddevs (np.array): output std predictions (train set)
        y_test_stddevs (np.array): output std predictions (test set)
        ci (float): confidence level 0.0 to 1.0

    Returns:
        Tuple: percentages of true outputs within the bounds of the
        confidence interval of the predicted density for both training and testing set
    """
    assert ci >= 0.0 and ci <= 1.0, "Confidence level \'ci\' must be betwen 0 and 1"
    z_value = stats.norm.ppf((1 + ci) / 2) 
    train_lower_bound = y_train_pred - z_value * y_train_stddevs
    train_upper_bound = y_train_pred + z_value * y_train_stddevs

    test_lower_bound = y_test_pred - z_value * y_test_stddevs
    test_upper_bound = y_test_pred + z_value * y_test_stddevs

    train_within_interval = np.sum(np.logical_and(y_train.ravel() >= train_lower_bound, y_train.ravel() <= train_upper_bound))
    test_within_interval = np.sum(np.logical_and(y_test.ravel() >= test_lower_bound, y_test.ravel() <= test_upper_bound))

    train_percentage_within_interval = (train_within_interval / len(y_train.ravel())) * 100
    test_percentage_within_interval = (test_within_interval / len(y_test.ravel())) * 100
    return train_percentage_within_interval, test_percentage_within_interval

    
def evaluate_and_save_metrics(model_name, y_train, y_test, y_train_pred, y_test_pred, y_train_stddevs=None, y_test_stddevs=None, ci=0.99, ci2=0.95, output_file="results.csv"):
    """Evaluates the specified predictions and saves the metrics' results into the output_file, if specified.

    Args:
        model_name (str): model name
        y_train (np.array): output true values (train set)
        y_test (np.array): output true values (test set)
        y_train_pred (np.array): output mean predictions (train set)
        y_test_pred (np.array): output mean predictions (test set)
        y_train_stddevs (np.array, optional): output std predictions (train set). Defaults to None.
        y_test_stddevs (np.array, optional): output std predictions (test set). Defaults to None.
        ci (float, optional): confidence level 0.0 to 1.0. Defaults to 0.99.
        ci2 (float, optional): confidence level 0.0 to 1.0. Defaults to 0.95.
        output_file (str, optional): file where to save the metrics. Defaults to "results.csv".
    """
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    train_mae = mean_absolute_error(y_train, y_train_pred)    
    test_mae = mean_absolute_error(y_test, y_test_pred) 

    train_percentage_within_interval, test_percentage_within_interval = "-", "-"
    train_percentage_within_interval2, test_percentage_within_interval2 = "-", "-"

    if y_train_stddevs is not None and y_test_stddevs is not None:
        train_percentage_within_interval, test_percentage_within_interval = calc_percentages_in_interval(y_train, y_test, y_train_pred, y_test_pred, y_train_stddevs, y_test_stddevs, ci)
        train_percentage_within_interval2, test_percentage_within_interval2 = calc_percentages_in_interval(y_train, y_test, y_train_pred, y_test_pred, y_train_stddevs, y_test_stddevs, ci2)

    # Print metrics
    print(f"Train RMSE: {train_rmse:.3f}")
    print(f"Test RMSE: {test_rmse:.3f}")
    print(f"Train MAE: {train_mae:.3f}")
    print(f"Test MAE: {test_mae:.3f}")

    print(f"Percentage of Test Data Points within {ci*100:.2f}% CI: " +
          (f"{train_percentage_within_interval}%" if isinstance(train_percentage_within_interval, str) else f"{train_percentage_within_interval:.2f}%"))
    print(f"Percentage of Test Data Points within {ci*100:.2f}% CI: " +
          (f"{test_percentage_within_interval}%" if isinstance(test_percentage_within_interval, str) else f"{test_percentage_within_interval:.2f}%"))
    print(f"Percentage of Test Data Points within {ci*100:.2f}% CI: " +
          (f"{train_percentage_within_interval2}%" if isinstance(train_percentage_within_interval2, str) else f"{train_percentage_within_interval2:.2f}%"))
    print(f"Percentage of Test Data Points within {ci*100:.2f}% CI: " +
          (f"{test_percentage_within_interval2}%" if isinstance(test_percentage_within_interval2, str) else f"{test_percentage_within_interval2:.2f}%"))
    
    # Generate new row of the dataframe
    if model_name is not None:
        new_row = pd.DataFrame({
            "Model Name": [model_name],
            "Train RMSE": [round(train_rmse, 2)],
            "Train MAE": [round(train_mae, 2)],
            "Test RMSE": [round(test_rmse, 2)],
            "Test MAE": [round(test_mae, 2)],
            f"Test % within {ci*100:.2f}% CI": [test_percentage_within_interval if isinstance(test_percentage_within_interval, str) \
                                                else round(test_percentage_within_interval, 2)],
            f"Test % within {ci2*100:.2f}% CI": [test_percentage_within_interval2 if isinstance(test_percentage_within_interval2, str) \
                                                else round(test_percentage_within_interval2, 2)]
        })

    # Update the output_file with the new row, either appending or overwriting
    try:
        results_df = pd.read_csv(output_file)
    except FileNotFoundError or EmptyDataError:
        results_df = pd.DataFrame(columns=list(new_row.columns))
    if model_name in results_df["Model Name"].values:
        results_df.loc[results_df["Model Name"] == model_name] = new_row.values
    elif model_name is not None:
        results_df = pd.concat([results_df, new_row], ignore_index=True)
    results_df.to_csv(output_file, index=False)
    
    
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
    plt.legend()
    red_patch = mpatches.Patch(color='red', alpha=0.3, label=f'Gap (MCE={max(abs(differences)):.2f})')
    handles.append(red_patch)
    plt.legend(handles=handles)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.show()


def save_preds(name, y_test_pred, y_test_stddevs, filename="preds.csv"):
    """Save predictions to CSV file

    Args:
        name (str): model name
        y_test_pred (np.array): test mean predictions
        y_test_stddevs (np.array): test standatd deviation predictions
        filename (str, optional): prediction file names. Defaults to "preds.csv".

    Returns:
        pd.DataFrame: dataframe in the prediciton CSV file
    """
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


def get_live_data(df, start_index=-144, end_index=None):
    df_live = df.loc[start_index:end_index]

    X, y = df_live[FEATURES].to_numpy(), df_live[OUTPUT_FEATURE].to_numpy()

    scaler = load('saved_models/scaler.joblib')
    X = scaler.transform(X)
    X.shape
    return X, y

def qq_plot(normalized_residuals, ci=0.99):
    """Plots the QQ plot of the normalised residuals

    Args:
        normalized_residuals (_type_): _description_
        ci (float, optional): _description_. Defaults to 0.99.
    """
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)

    z_value = stats.norm.ppf((1 + ci) / 2)

    # Calculate theoretical quantiles and observed quantiles
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.5/len(normalized_residuals), 1-0.5/len(normalized_residuals), len(normalized_residuals)))
    observed_quantiles = np.sort(normalized_residuals)
    lower_bound = 0 - z_value 
    upper_bound = 0 + z_value

    outliers_mask = (observed_quantiles < lower_bound) | (observed_quantiles > upper_bound)
    ax.plot(theoretical_quantiles[outliers_mask], observed_quantiles[outliers_mask], 'o', color='red')
    ax.plot(theoretical_quantiles[~outliers_mask], observed_quantiles[~outliers_mask], 'o', color='black')

    slope, intercept, _, _, _ = stats.linregress(theoretical_quantiles, observed_quantiles)
    line = slope * theoretical_quantiles + intercept
    ax.plot(theoretical_quantiles, line, color='gray', linestyle='--', label=f'Fit: slope={slope:.2f},\nintercept={intercept:.2f}')

    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Observed Quantiles')
    plt.show()


def update_plot_wrapper(df, model):
    def update_plot_lambda(start_index, end_index, ci):
        update_plot(df, model, start_index, end_index, ci)
    return update_plot_lambda


def update_plot(df, model, start_index, end_index, ci):
    """Update qq plot with new predictions

    Args:
        df (pd:DataFrame): dataframe
        model (keras.Model): model
        start_index (int): start index
        end_index (int): end index
        ci (float): confidence level
    """
    if end_index <= start_index:
        clear_output(wait=True)
        print("Error: End index must be greater than start index.")
        return
    
    clear_output(wait=True)
    
    X_live, y_live = get_live_data(df, start_index, end_index)
    
    y_live_pred = np.array(model(X_live).mean()).ravel()
    y_live_stddevs = np.array(model(X_live).stddev()).ravel()  
    
    normalized_residuals = (y_live - y_live_pred) / y_live_stddevs
    
    qq_plot(normalized_residuals, ci)


def interactive_qq_plot(df, model):
    start_index_widget = widgets.IntText(value=230, description='Start Index:', continuous_update=False)
    end_index_widget = widgets.IntText(value=374, description='End Index:', continuous_update=False)
    ci_widget = widgets.FloatSlider(value=0.99, min=0.90, max=0.999, step=0.001, description='Confidence Interval:', continuous_update=False)
    
    update_plot_func = update_plot_wrapper(df, model)
    interact_manual = widgets.interactive(update_plot_func, start_index=start_index_widget, end_index=end_index_widget, ci=ci_widget)
    
    display(interact_manual)


def calculate_normalized_residuals(model, X, y):
    y_pred = np.array(model(X).mean()).ravel()
    y_stddevs = np.array(model(X).stddev()).ravel()  
    return (y - y_pred) / y_stddevs


def detect_visible_faults(df, mask_fault, mask_period=None):
    plt.figure(figsize=(10, 6))

    plt.scatter(df['Wind.speed.me'], df[OUTPUT_FEATURE], color='0.3', alpha=0.7, linewidth=0, s=2, label='All Data Points')

    red_points = df.loc[mask_fault]
    plt.scatter(red_points['Wind.speed.me'], red_points[OUTPUT_FEATURE], color='red', marker='x', s=100, label='Selected Points')
    if mask_period is not None:
        yellow_points = df.loc[mask_period]
        plt.scatter(yellow_points['Wind.speed.me'], yellow_points[OUTPUT_FEATURE], color='orange', alpha=0.5, label='143 Points Before')

    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Wind Power (kW)')
    plt.legend()
    plt.grid(True)
    plt.show()
    indices = red_points
    if mask_period is not None:
        indices = df.loc[mask_fault | mask_period]
    return indices


def cusum_test_plot(residuals, datetime_values, target=0, k=0.5, h=5, save_path=None):
    """Perform a two-sided CUSUM test on residuals and plot.

    Args:
       residuals (list or array-like): List of normalized residuals.
        datetime_values (list or array-like): List of datetime values corresponding to the residuals.
        target (float): Target mean for normalized residuals.
        k (float): Reference value (allowable slack before signal), typically a small positive number.
        h (float): Decision interval (control limit).
    """
    datetime_values.reset_index(drop=True, inplace=True)  # Reset index

    colors = ['#445469', '#772E15']
    S_pos = [0]
    S_neg = [0]

    # Two-sided CUSUM test
    for i in range(1, len(residuals)):
        S_pos.append(max(0, S_pos[i-1] + residuals[i] - target - k))
        S_neg.append(min(0, S_neg[i-1] + residuals[i] - target + k))
        
    # Find the index of the first occurrence where the control limit is surpassed
    control_index_pos = next((i for i, value in enumerate(S_pos) if value > h), None)
    control_index_neg = next((i for i, value in enumerate(S_neg) if value < -h), None)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(datetime_values, S_pos, label='S' + '\u2095', color=colors[0])
    plt.plot(datetime_values, S_neg, label='S' + '\u2097', color=colors[1])
    plt.axhline(y=h, color=colors[0], linestyle='--', label='-I')
    plt.axhline(y=-h, color=colors[1], linestyle='--', label='I')
    plt.xlabel('Date & Time')
    plt.ylabel('Cumulative Sum')
    plt.grid(True)

    y_min = -10
    y_max = 10
    plt.ylim(-h * 1.5, h * 1.5)

    if min(S_neg) < y_min:
        plt.ylim(y_min * 1.1, y_max)
    if max(S_pos) > y_max:
        plt.ylim(y_min, y_max * 1.1)

    indices = np.linspace(0, len(datetime_values) - 1, 5, dtype=int)
    selected_dates = pd.to_datetime(datetime_values.iloc[indices])

    plt.xticks(selected_dates, [date.strftime('%Y-%m-%d\n%H:%M') for date in selected_dates], rotation=45, ha='right')

    if control_index_pos is not None:
        plt.scatter(datetime_values.iloc[control_index_pos], S_pos[control_index_pos], color='none', edgecolor='red', linewidths=2, s=200, zorder=5)
        plt.text(datetime_values.iloc[control_index_pos], S_pos[control_index_pos] + 0.75, 
                 datetime_values.iloc[control_index_pos].strftime('%Y-%m-%d\n%H:%M'), 
                 ha='right', fontsize=12, color='black', va='top', zorder=6,
                 bbox=dict(boxstyle="round", ec='black', fc='white', alpha=0.5))
    if control_index_neg is not None:
        plt.scatter(datetime_values.iloc[control_index_neg], S_neg[control_index_neg], color='none', edgecolor='red', linewidths=2, s=200, zorder=5)
        plt.text(datetime_values.iloc[control_index_neg], S_neg[control_index_neg] - 0.75, 
                 datetime_values.iloc[control_index_neg].strftime('%Y-%m-%d\n%H:%M'), 
                 ha='right', fontsize=12, color='black', va='top', zorder=6,
                 bbox=dict(boxstyle="round", ec='black', fc='white', alpha=0.5))
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()


def plot_calibration_errors(y_test_pred_list, y_test_std_list, y_test_list, bins=20, titles=None, with_legend=False, save_path=None):
    """Plots the calibration errors

    Args:
        y_test_pred_list (_type_): list of predicted test means
        y_test_std_list (_type_): list of predicted test standard deviations
        y_test_list (_type_): list of true test values
        bins (int, optional): number of bins. Defaults to 20.
        titles (_type_, optional): titles. Defaults to None.
        save_path (_type_, optional): file path to save to. Defaults to None.
    """
    plt.rc('font', size=18)
    fig, ax = plt.subplots(figsize=(12, 8))

    p_values = np.array([stats.norm.ppf(i/bins + (1-i/bins)/2) for i in range(1, bins)]).squeeze()
    t_values = np.concatenate((p_values, np.array([stats.norm.ppf(0.995)])))
    
    markers = ['o', 's', '^', '>', '<', '*']
    colors = ['#DC653D', '#222A35', '#445469', '#8497B0', '#772E15', '#52883B']

    assert len(y_test_pred_list) == len(markers) and len(y_test_std_list) == len(markers) and len(y_test_list) == len(markers), \
        "prediction lists and colors should have the same length"

    ax.plot([0, 100], [0, 0], color='red', linestyle='--')

    for i, (y_test_pred, y_test_std, y_test) in enumerate(zip(y_test_pred_list, y_test_std_list, y_test_list)):
        actual_percentages = np.concatenate((np.arange(1, bins)*100/bins, np.array([99])))
        percentages_within_interval = []
        for t_value in t_values:
            lower_bounds = y_test_pred.ravel() - t_value * y_test_std
            upper_bounds = y_test_pred.ravel() + t_value * y_test_std

            is_within_interval = np.logical_and(y_test >= lower_bounds, y_test <= upper_bounds)
            num_within_interval = np.sum(is_within_interval)
            percentage_within_interval = (num_within_interval / len(y_test)) * 100
            percentages_within_interval.append(percentage_within_interval)
        calibration_error = np.array(percentages_within_interval) - actual_percentages
        print(max(abs(calibration_error)))
        ax.scatter(actual_percentages, calibration_error, marker=markers[i], color=colors[i], label=titles[i], s=100)

    ax.set_xlim(0, 105)
    ax.set_xticks(list(range(0, 91, 10)) + [99])
    ax.set_xlabel('% Confidence Interval', fontsize=18)
    ax.set_ylabel('Calibration Error', fontsize=18)

    plt.subplots_adjust(bottom=0.2)
    if with_legend:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3, fontsize=18)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    if with_legend:
        plt.show()