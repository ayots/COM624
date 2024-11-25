import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objects as go

# Step 0: File Upload and Basic Validation
st.title("Personal Finance Management System with LSTM and Interactive Features")

uploaded_file = st.file_uploader("Upload your CSV file:", type=["csv"])
if uploaded_file is not None:
    try:
        # Read uploaded CSV file
        data = pd.read_csv(uploaded_file)
        st.write("Data Preview:", data.head())
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
else:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

# Step 1: Data Cleaning and Preprocessing
st.subheader("Step 1: Data Cleaning and Preprocessing")

# Define target column
target_col = 'Savings for Property (£)'

try:
    # Drop non-numeric columns
    non_numeric_columns = data.select_dtypes(exclude=["number"]).columns
    if non_numeric_columns.size > 0:
        st.write("Dropping non-numeric columns:", non_numeric_columns.tolist())
        data = data.drop(columns=non_numeric_columns)

    # Check if the target column exists
    if target_col not in data.columns:
        st.error(f"Required column '{target_col}' is missing in the dataset.")
        st.stop()

    # Drop rows with missing target values and fill remaining NaNs with column means
    data_cleaned = data.dropna(subset=[target_col])
    data_cleaned.fillna(data_cleaned.mean(), inplace=True)

    st.write("Cleaned Data Preview:", data_cleaned.head())
except Exception as e:
    st.error(f"Data cleaning error: {e}")
    st.stop()

# Step 2: Data Visualization
st.subheader("Step 2: Data Visualization")
try:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(data_cleaned)), y=data_cleaned[target_col], mode='lines', name=target_col))
    fig.update_layout(title="Savings for Property Over Time", xaxis_title="Index", yaxis_title="Savings (£)")
    st.plotly_chart(fig)
except Exception as e:
    st.error(f"Data visualization error: {e}")
    st.stop()

# Step 3: LSTM Model for Forecasting
st.subheader("Step 3: LSTM Model for Forecasting")
try:
    # Scale the data
    scaler = MinMaxScaler()
    data_cleaned_scaled = scaler.fit_transform(data_cleaned[[target_col]])

    # Create sequences for LSTM
    sequence_length = 10
    X, y = [], []
    for i in range(len(data_cleaned_scaled) - sequence_length):
        X.append(data_cleaned_scaled[i:i+sequence_length])
        y.append(data_cleaned_scaled[i+sequence_length])

    X, y = np.array(X), np.array(y)

    # Split into training and testing sets
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Define LSTM model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    st.write("Training LSTM model...")
    model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=0)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_rescaled = scaler.inverse_transform(y_pred)
    y_test_rescaled = scaler.inverse_transform(y_test)

    # Plot predictions vs actual
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_test_rescaled.flatten(), mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(y=y_pred_rescaled.flatten(), mode='lines', name='Predicted'))
    fig.update_layout(title="LSTM Predictions vs Actual", xaxis_title="Index", yaxis_title="Savings (£)")
    st.plotly_chart(fig)
except Exception as e:
    st.error(f"Model training/prediction error: {e}")
    st.stop()

# Step 4: Decision-Making Support
st.subheader("Step 4: Decision-Making Support")
try:
    current_savings = data_cleaned[target_col].iloc[-1]  # Ensure data_cleaned is defined before this step
    interval = st.selectbox("Select Interval:", ["Daily", "Weekly", "Monthly"])
    savings_goal = st.number_input("Set Your Savings Target (£):", min_value=0.0, step=100.0)

    if current_savings < savings_goal:
        st.warning(f"You need to save an additional £{savings_goal - current_savings:.2f} to meet your target.")
    else:
        st.success("Congratulations! You have met your savings goal.")
except Exception as e:
    st.error(f"Decision-making support error: {e}")

# Step 5: Interactivity and Real-Time Updates
st.subheader("Step 5: Interactivity and Real-Time Updates")
try:
    real_time_savings = st.slider("Adjust Current Savings (£):", min_value=0, max_value=int(current_savings + 5000), value=int(current_savings))
    updated_goal_status = "Met" if real_time_savings >= savings_goal else "Not Met"
    st.write(f"Updated Goal Status: {updated_goal_status}")
except Exception as e:
    st.error(f"Real-time updates error: {e}")

# Step 6: Scenario Planning and Forecasting
st.subheader("Step 6: Scenario Planning and Forecasting")
try:
    scenario_increase = st.number_input("Increase Savings by (%):", min_value=0, max_value=100, step=5)
    forecasted_savings = real_time_savings * (1 + scenario_increase / 100)
    st.write(f"If you increase savings by {scenario_increase}%, your forecasted savings will be £{forecasted_savings:.2f}.")
except Exception as e:
    st.error(f"Scenario planning error: {e}")
