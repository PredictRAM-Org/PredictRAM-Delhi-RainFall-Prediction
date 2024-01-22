import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load the data
file_path = 'RainPredictionFileWithPercentage.xlsx'
df = pd.read_excel(file_path)

# Apply rain prediction conditions
humidity_condition = (df['humidity'] >= 40) & (df['humidity'] <= 60)
windspeed_condition = (df['windspeedKmph'] >= 9) & (df['windspeedKmph'] <= 15)

rain_condition = (df['maxtempC'] > 33) & (df['mintempC'] > 25) & humidity_condition & windspeed_condition

df['RainPrediction'] = 'No'
df.loc[rain_condition, 'RainPrediction'] = 'Yes'

# Encode 'RainPrediction' labels to numeric values
label_encoder = LabelEncoder()
df['RainPrediction'] = label_encoder.fit_transform(df['RainPrediction'])

# Select features and target
selected_features = ["maxtempC", "mintempC", "humidity", "winddirDegree", "windspeedKmph"]
X = df[selected_features]
y = df['RainPrediction']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a Random Forest Classifier in a pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values by imputing the mean
    ('scaler', StandardScaler()),  # Standardize numerical features
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Streamlit app
st.title("Rain Prediction App")

maxtemp = st.number_input("Max Temperature (°C)")
mintemp = st.number_input("Min Temperature (°C)")
humidity = st.number_input("Humidity (%)")
winddir_degree = st.number_input("Wind Direction Degree")
windspeed = st.number_input("Wind Speed (Kmph)")

if st.button("Predict"):
    # Make predictions using the trained model
    new_data = pd.DataFrame({
        'maxtempC': [maxtemp],
        'mintempC': [mintemp],
        'humidity': [humidity],
        'winddirDegree': [winddir_degree],
        'windspeedKmph': [windspeed]
    })

    prediction = pipeline.predict(new_data)[0]

    # Convert numeric prediction back to 'Yes' or 'No'
    prediction_label = label_encoder.inverse_transform([prediction])[0]

    st.write(f"Rain Prediction: {prediction_label}")
