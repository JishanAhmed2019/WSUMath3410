import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
import pandas as pd
# Set page configuration
st.set_page_config(page_title='Stock Analysis App', layout='wide', initial_sidebar_state='expanded')
# Function to load data
def load_data(ticker, years):
    end_date = pd.to_datetime('today')
    start_date = end_date - pd.DateOffset(years=years)
    return yf.download(ticker, start=start_date, end=end_date)

# Function to check stationarity
def check_stationarity(data):
    adf_test = adfuller(data['Close'])
    return adf_test[1] > 0.05

# Function to plot data
def plot_data(data, title, ylabel, color, diff=False):
    plt.style.use('seaborn-v0_8-bright')
    plt.figure(figsize=(12, 6))
    if diff:
        plt.plot(data['Diff'], label='Differenced Close Price', color=color)
    else:
        plt.plot(data['Close'], label='Close Price', color=color)
    plt.title(title, fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# Sidebar for user inputs
st.sidebar.header('User Input Parameters')
ticker = st.sidebar.text_input('Stock Ticker Symbol (e.g., GME):')
years = st.sidebar.slider('Number of Years of Data:', 1, 20, 5)
# Footer for the signature
footer = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: white;
    color: crimson;
    text-align: center;
    padding: 10px;
    font-size: 16px;
}
</style>
<div class='footer'>
     <p><b>Created by Dr. Jishan Ahmed</b></p>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)   
st.markdown("<h1 style='color: darkred;'>Time Series Analysis and Forecasting of Stock Price</h1>", unsafe_allow_html=True)

# Main section
#st.title('Stock Time Series Analysis and Forecasting')

# Display user inputs
st.write(f'Analyzing {years} years of stock data for ticker symbol: {ticker}')

if ticker:
    # Load and plot the data
    data = load_data(ticker, years)
    st.markdown(f"<h2 style='color: blue;'>Closing Prices of {ticker} Stock</h2>", unsafe_allow_html=True)
    #st.subheader(f'{ticker} Stock Closing Prices')
    plot_data(data, f'{ticker} Stock Closing Prices', 'Price (USD)', 'blue')

    # Check for stationarity
    st.markdown("<h2 style='color:blue;'>Stationarity Check and Data Transformation</h2>", unsafe_allow_html=True)
    #st.subheader('Stationarity Check and Data Transformation')
    if check_stationarity(data):
        st.write('Data is not stationary. Differencing data...')
        data['Diff'] = data['Close'].diff().dropna()
        plot_data(data, f'Differenced {ticker} Stock Closing Prices', 'Differenced Price', 'green', diff=True)
    else:
        st.write('Data is stationary. No transformation required.')

    # Fit AutoARIMA model and forecast
    st.markdown("<h2 style='color:orangered;'>Model Training</h2>", unsafe_allow_html=True)
    #st.subheader('Forecasting Stock Prices')
    st.write('Fitting AutoARIMA model...')
    model = auto_arima(data['Close'], seasonal=False, trace=False)
    n_periods = 30  # Forecast the next 30 days
    forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)
    future_dates = pd.date_range(data.index[-1], periods=n_periods, freq='B')

    # Plotting forecast
    st.markdown("<h2 style='color:magenta;'>Forecasting Stock Prices</h2>", unsafe_allow_html=True)
    #st.subheader('Forecast Results')
    plt.style.use('seaborn-v0_8-bright')
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='Historical Close Price', color='blue')
    plt.plot(future_dates, forecast, label='Forecast', color='red')
    plt.fill_between(future_dates, conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3)
    plt.title(f'{ticker} Stock Price Forecast', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
else:
    st.write('Please enter a valid stock ticker symbol in the sidebar.')


