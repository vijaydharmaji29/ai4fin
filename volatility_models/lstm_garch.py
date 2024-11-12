import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from arch import arch_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import csv

def train(file):
    # Load your time series data (e.g., stock returns)
    df = pd.read_csv(file)

    # Sort the data by date (assuming there's a 'date' column)
    df['date'] = pd.to_datetime(df['Reported Date'])  # Ensure date column is in datetime format
    df = df.sort_values(by='date')

    # Preprocess data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Modal Price (Rs./Quintal)'].values.reshape(-1, 1))

    # Prepare the data for LSTM (creating sequences)
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length, 0])
            y.append(data[i + seq_length, 0])
        return np.array(X), np.array(y)

    SEQ_LENGTH = 60  # Number of time steps to consider
    X, y = create_sequences(scaled_data, SEQ_LENGTH)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Split data into training and testing sets
    split_ratio = 0.8
    split_idx = int(len(X) * split_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))  # Output layer

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the LSTM model
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

    # Get LSTM predictions and residuals (errors)
    lstm_predictions = model.predict(X_test)
    lstm_predictions = scaler.inverse_transform(lstm_predictions)

    # Calculate residuals from LSTM predictions
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    residuals = y_test_actual[:, 0] - lstm_predictions[:, 0]

    # Fit GARCH model on LSTM residuals
    garch_model = arch_model(residuals, vol='Garch', p=1, q=1)
    garch_fit = garch_model.fit(disp="off")

    # Predict volatility using the GARCH model
    garch_volatility = garch_fit.forecast(horizon=1).variance[-1:]

    # Combine LSTM and GARCH results (final volatility prediction)
    final_predictions = lstm_predictions + np.sqrt(garch_volatility.values)

    # Calculate Accuracy Metrics
    mae = mean_absolute_error(y_test_actual, final_predictions)
    mse = mean_squared_error(y_test_actual, final_predictions)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test_actual - final_predictions) / y_test_actual)) * 100

    # Print the results
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape}%")

    accuracy_results = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    }
    
    return accuracy_results


if __name__ == '__main__':
    files = os.listdir('../ai4fin/data/')
    results = {
        'Commodity': [],
        'MAE': [],
        'MSE': [],
        'RMSE': [],
        'MAPE': []
    }
    for file in files:
        if file[-3:] != 'csv':
            continue
        com_results = train('../ai4fin/data/' + file)
        results['Commodity'].append(file.split('_')[0])
        results['MAE'].append(com_results['MAE'])
        results['MSE'].append(com_results['MSE'])
        results['RMSE'].append(com_results['RMSE'])
        results['MAPE'].append(com_results['MAPE'])

    csv_file = "lstm_garch_results.csv"

    # Write the dictionary to CSV
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=results.keys())
        
        # Write the header (column names)
        writer.writeheader()
        
        # Write the rows (zip the values to create rows)
        for row in zip(*results.values()):
            writer.writerow(dict(zip(results.keys(), row)))

    print("DONE")
    
