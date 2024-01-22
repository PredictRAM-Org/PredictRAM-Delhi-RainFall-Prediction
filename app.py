# app.py
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load the data
file_path = 'RainPredictionFileWithPercentage.xlsx'
df = pd.read_excel(file_path)

# Select features and target
selected_features = ["maxtempC", "mintempC", "humidity", "windspeedKmph"]
X = df[selected_features]
y = df["RainPercentage"]

# Scale the features for LSTM
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Build Linear Regression model
linear_reg_model = LinearRegression()
linear_reg_model.fit(X, y)

# Build LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, activation='relu', input_shape=(1, X_scaled.shape[1])))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_lstm, y, epochs=50, batch_size=32, verbose=0)

def predict_linear_regression(maxtemp, mintemp, humidity, windspeed):
    return linear_reg_model.predict([[maxtemp, mintemp, humidity, windspeed]])[0]

def predict_lstm(maxtemp, mintemp, humidity, windspeed):
    new_data_scaled = scaler.transform([[maxtemp, mintemp, humidity, windspeed]])
    new_data_lstm = new_data_scaled.reshape((1, 1, len(selected_features)))
    return lstm_model.predict(new_data_lstm)[0][0]

# Streamlit app
st.title("Rain Prediction App")

maxtemp = st.number_input("Max Temperature (°C)")
mintemp = st.number_input("Min Temperature (°C)")
humidity = st.number_input("Humidity (%)")
windspeed = st.number_input("Wind Speed (Kmph)")

if st.button("Predict"):
    linear_reg_prediction = predict_linear_regression(maxtemp, mintemp, humidity, windspeed)
    lstm_prediction = predict_lstm(maxtemp, mintemp, humidity, windspeed)

    st.write(f"Linear Regression Prediction: {linear_reg_prediction}")
    st.write(f"LSTM Prediction: {lstm_prediction}")
