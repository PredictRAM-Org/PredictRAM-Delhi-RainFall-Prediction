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

# Apply rain prediction conditions
humidity_condition = (df['humidity'] >= 40) & (df['humidity'] <= 60)
windspeed_condition = (df['windspeedKmph'] >= 9) & (df['windspeedKmph'] <= 15)

rain_condition = (df['maxtempC'] > 33) & (df['mintempC'] > 25) & humidity_condition & windspeed_condition

df.loc[rain_condition, 'RainPrediction'] = 'Yes'

# Calculate rain fall percentage based on humidity and windspeed
df['RainPercentage'] = 0  # Default value

humidity_factor = (df['humidity'] - 40) / (60 - 40)
windspeed_factor = (df['windspeedKmph'] - 9) / (15 - 9)

rain_percentage = 10 + 60 * (humidity_factor + windspeed_factor) / 2
df.loc[rain_condition, 'RainPercentage'] = rain_percentage

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
    if (maxtemp > 33) and (mintemp > 25) and (humidity >= 40) and (humidity <= 60) and (windspeed >= 9) and (windspeed <= 15):
        return 'Yes', 10 + 60 * ((humidity - 40) / 20 + (windspeed - 9) / 6) / 2
    else:
        return 'No', 0

def predict_lstm(maxtemp, mintemp, humidity, windspeed):
    new_data_scaled = scaler.transform([[maxtemp, mintemp, humidity, windspeed]])
    new_data_lstm = new_data_scaled.reshape((1, 1, len(selected_features)))
    lstm_prediction = lstm_model.predict(new_data_lstm)[0][0]
    return 'Yes' if lstm_prediction > 0 else 'No', lstm_prediction

# Streamlit app
st.title("Rain Prediction App")

maxtemp = st.number_input("Max Temperature (°C)")
mintemp = st.number_input("Min Temperature (°C)")
humidity = st.number_input("Humidity (%)")
windspeed = st.number_input("Wind Speed (Kmph)")

if st.button("Predict"):
    linear_reg_prediction, linear_reg_percentage = predict_linear_regression(maxtemp, mintemp, humidity, windspeed)
    lstm_prediction, lstm_percentage = predict_lstm(maxtemp, mintemp, humidity, windspeed)

    st.write(f"Linear Regression Prediction: {linear_reg_prediction}")
    st.write(f"Linear Regression Percentage: {linear_reg_percentage}")

    st.write(f"LSTM Prediction: {lstm_prediction}")
    st.write(f"LSTM Percentage: {lstm_percentage}")
