import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

def prepare_data(stock_code):
    engine = create_engine('sqlite:///retail_data.db')
    data = pd.read_sql_query(f"""
        SELECT InvoiceDate, SUM(Quantity) as TotalQuantity
        FROM clean_transactions
        WHERE StockCode = '{stock_code}'
        GROUP BY InvoiceDate
        ORDER BY InvoiceDate
    """, engine)
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    data.set_index('InvoiceDate', inplace=True)
    return data

def arima_model(data):
    model = ARIMA(data['TotalQuantity'], order=(1, 1, 1))
    results = model.fit()
    forecast = results.forecast(steps=15)
    return forecast

def ets_model(data):
    model = ExponentialSmoothing(data['TotalQuantity'], trend='add', seasonal='add', seasonal_periods=7)
    results = model.fit()
    forecast = results.forecast(15)
    return forecast

def prophet_model(data):
    df = data.reset_index().rename(columns={'InvoiceDate': 'ds', 'TotalQuantity': 'y'})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=15)
    forecast = model.predict(future)
    return forecast['yhat'].tail(15)

def lstm_model(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data['TotalQuantity'].values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60, 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, batch_size=1, epochs=1)
    
    last_60_days = scaled_data[-60:]
    X_test = []
    for i in range(15):
        X_test.append(last_60_days[i:i+60])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    return predictions.flatten()

def evaluate_models(data, forecasts):
    actual = data['TotalQuantity'].values[-15:]
    for model_name, forecast in forecasts.items():
        mse = mean_squared_error(actual, forecast)
        mae = mean_absolute_error(actual, forecast)
        print(f"{model_name} - MSE: {mse:.2f}, MAE: {mae:.2f}")

# Example usage
stock_code = '85123A'  # Replace with one of your top 10 stock codes
data = prepare_data(stock_code)

forecasts = {
    'ARIMA': arima_model(data),
    'ETS': ets_model(data),
    'Prophet': prophet_model(data),
    'LSTM': lstm_model(data)
}

evaluate_models(data, forecasts)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['TotalQuantity'], label='Actual')
for model_name, forecast in forecasts.items():
    plt.plot(pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=15), forecast, label=f'{model_name} Forecast')
plt.legend()
plt.title(f'Demand Forecast for Stock Code {stock_code}')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.savefig(f'forecast_{stock_code}.png')
plt.close()



from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def plot_acf_pacf(data, lags=40):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    plot_acf(data, lags=lags, ax=ax1)
    plot_pacf(data, lags=lags, ax=ax2)
    plt.savefig('acf_pacf_plot.png')
    plt.close()

# Example usage
stock_code = '85123A'  # Replace with one of your top 10 stock codes
data = prepare_data(stock_code)
plot_acf_pacf(data['TotalQuantity'])