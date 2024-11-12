import numpy as np
import pandas as pd
from arch import arch_model
from pyswarm import pso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import os

# Define the directory containing your data files
data_directory = '../ai4fin/data/'  # Change this to the path of your data directory

# Function to standardize data
def standardize_data(returns):
    scaler = StandardScaler()
    returns_standardized = scaler.fit_transform(returns.values.reshape(-1, 1))
    return pd.Series(returns_standardized.flatten(), index=returns.index)

# Function to run models and calculate metrics
def process_file(file_path):
    # Load the data
    data = pd.read_csv(file_path)
    returns = pd.Series(data['Modal Price (Rs./Quintal)'])  # Assuming the column is named 'returns'
    
    # Standardize the returns
    returns_standardized = standardize_data(returns)

    # Run volatility models
    garch_result = run_garch(returns_standardized)
    gjr_result = run_gjr(returns_standardized)
    ewma_result = run_ewma(returns_standardized)
    mem_result = run_mem(returns_standardized)
    p_wev_result = run_p_wev(returns_standardized)

    # Create DataFrame for results
    result_df = pd.DataFrame({
        'GARCH': garch_result.conditional_volatility,
        'GJR-GARCH': gjr_result.conditional_volatility,
        'EWMA': ewma_result,
        'MEM': mem_result,
        'P-WEV': p_wev_result
    })

    # Calculate accuracy metrics
    true_vol = returns_standardized.rolling(window=2).std().dropna()  # Using rolling std as a proxy for true volatility
    metrics = {}

    for model_name, vol_series in result_df.items():
        vol_series = vol_series[true_vol.index]  # Align indices
        rmse, mae, mape, r2 = calculate_metrics(true_vol, vol_series)
        metrics[model_name] = [rmse, mae, mape, r2]

    return metrics

# Define models and metrics
def run_garch(returns):
    garch = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
    garch_result = garch.fit(disp='off')
    return garch_result

def run_gjr(returns):
    gjr = arch_model(returns, vol='Garch', p=1, q=1, o=1, rescale=False)
    gjr_result = gjr.fit(disp='off')
    return gjr_result

def run_ewma(returns, lambda_value=0.94):
    ewma_vol = [np.std(returns)]
    for i in range(1, len(returns)):
        ewma_vol.append(np.sqrt((1 - lambda_value) * returns[i-1]**2 + lambda_value * ewma_vol[i-1]**2))
    return pd.Series(ewma_vol, index=returns.index)

def run_mem(returns):
    alpha = 0.1
    mem_vol = np.zeros_like(returns)
    mem_vol[0] = np.std(returns)
    for i in range(1, len(returns)):
        mem_vol[i] = alpha * np.abs(returns[i-1]) + (1 - alpha) * mem_vol[i-1]
    return pd.Series(mem_vol, index=returns.index)

def p_wev_obj_function(weights, vol_matrix):
    vol = np.sqrt(np.dot(weights, np.var(vol_matrix, axis=0)))
    return vol

def run_p_wev(returns):
    garch_vol = run_garch(returns).conditional_volatility
    gjr_vol = run_gjr(returns).conditional_volatility
    ewma_vol = run_ewma(returns)
    mem_vol = run_mem(returns)
    
    vol_matrix = np.vstack([garch_vol, gjr_vol, ewma_vol, mem_vol]).T
    
    n_models = 4
    lb = [0] * n_models
    ub = [1] * n_models
    
    optimal_weights, _ = pso(p_wev_obj_function, lb, ub, args=(vol_matrix,))
    
    ensemble_vol = np.dot(vol_matrix, optimal_weights)
    
    return pd.Series(ensemble_vol, index=returns.index)

def calculate_metrics(true_values, predicted_values):
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    mae = mean_absolute_error(true_values, predicted_values)
    
    # Avoid division by zero by excluding zero values
    nonzero_indices = true_values != 0
    true_values_nonzero = true_values[nonzero_indices]
    predicted_values_nonzero = predicted_values[nonzero_indices]
    
    if len(true_values_nonzero) == 0:
        mape = np.nan  # Or another appropriate value if there are no non-zero values
    else:
        mape = np.mean(np.abs((true_values_nonzero - predicted_values_nonzero) / true_values_nonzero)) * 100
    
    r2 = r2_score(true_values, predicted_values)
    return rmse, mae, mape, r2

# Loop through all files and process
file_metrics = []
for file_name in os.listdir(data_directory):
    if file_name.endswith('.csv'):
        file_path = os.path.join(data_directory, file_name)
        metrics = process_file(file_path)
        for model_name, model_metrics in metrics.items():
            rmse, mae, mape, r2 = model_metrics
            file_metrics.append({
                'File': file_name,
                'Model': model_name,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape,
                'R2': r2
            })

# Convert results to DataFrame
metrics_df = pd.DataFrame(file_metrics)

# Save metrics to separate CSV files for each type of accuracy measurement
metrics_df[['File', 'Model', 'RMSE']].to_csv('rmse_results.csv', index=False)
metrics_df[['File', 'Model', 'MAE']].to_csv('mae_results.csv', index=False)
metrics_df[['File', 'Model', 'MAPE']].to_csv('mape_results.csv', index=False)
metrics_df[['File', 'Model', 'R2']].to_csv('r2_results.csv', index=False)

print("Metrics saved to 'rmse_results.csv', 'mae_results.csv', 'mape_results.csv', and 'r2_results.csv'")
