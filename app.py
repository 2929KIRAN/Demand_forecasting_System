import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from time_series_analysis import prepare_data, arima_model, ets_model, prophet_model, lstm_model

# Load top 10 products
top_10_quantity = pd.read_csv('top_10_quantity.csv')
top_10_revenue = pd.read_csv('top_10_revenue.csv')

st.title('Demand Forecasting System')

# Display top 10 products
st.subheader('Top 10 Products by Quantity Sold')
st.dataframe(top_10_quantity)

st.subheader('Top 10 Products by Revenue')
st.dataframe(top_10_revenue)

# User input
selected_stock = st.selectbox('Select a Stock Code', top_10_quantity['StockCode'].tolist())
forecast_weeks = st.slider('Number of Weeks to Forecast', 1, 15, 15)

if st.button('Generate Forecast'):
    # Prepare data
    data = prepare_data(selected_stock)
    
    # Generate forecasts
    forecasts = {
        'ARIMA': arima_model(data),
        'ETS': ets_model(data),
        'Prophet': prophet_model(data),
        'LSTM': lstm_model(data)
    }
    
    # Plot historical and forecasted demand
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['TotalQuantity'], label='Historical Demand')
    for model_name, forecast in forecasts.items():
        ax.plot(pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_weeks), 
                forecast[:forecast_weeks], label=f'{model_name} Forecast')
    ax.legend()
    ax.set_title(f'Demand Forecast for Stock Code {selected_stock}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Quantity')
    st.pyplot(fig)
    
    # Error histograms
    for model_name, forecast in forecasts.items():
        errors = data['TotalQuantity'].values[-forecast_weeks:] - forecast[:forecast_weeks]
        fig, ax = plt.subplots()
        ax.hist(errors, bins=20)
        ax.set_title(f'{model_name} Error Distribution')
        ax.set_xlabel('Error')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)
    
    # Download forecast as CSV
    forecast_df = pd.DataFrame(forecasts)
    forecast_df.index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_weeks)
    csv = forecast_df.to_csv(index=True)
    st.download_button(
        label="Download Forecast CSV",
        data=csv,
        file_name=f"forecast_{selected_stock}.csv",
        mime="text/csv"
    )

# Run the Streamlit app
if __name__ == '__main__':
    st.run()