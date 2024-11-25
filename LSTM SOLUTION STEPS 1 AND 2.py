# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Load the dataset
data = pd.read_csv("personal_finance_employees_V1.csv")  # Replace with your file path

# Data Preprocessing
data_cleaned = data.dropna(subset=['Savings for Property (£)'])  # Drop rows with missing target values
numeric_columns = data_cleaned.select_dtypes(include=[np.number]).columns  # Identify numeric columns
data_cleaned[numeric_columns] = data_cleaned[numeric_columns].fillna(data_cleaned[numeric_columns].mean())  # Fill missing numeric values

# Select features and target
features = data_cleaned[
    ['Monthly Income (£)', 'Electricity Bill (£)', 'Gas Bill (£)', 'Netflix (£)',
     'Amazon Prime (£)', 'Groceries (£)', 'Transportation (£)', 'Water Bill (£)',
     'Sky Sports (£)', 'Other Expenses (£)', 'Monthly Outing (£)']
]
target = data_cleaned['Savings for Property (£)']

# Normalize data
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
features_scaled = scaler_x.fit_transform(features)
target_scaled = scaler_y.fit_transform(target.values.reshape(-1, 1))

# Reshape data for LSTM input
X = features_scaled.reshape(features_scaled.shape[0], 1, features_scaled.shape[1])  # (samples, time steps, features)
y = target_scaled

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM Model
model = Sequential([
    LSTM(64, input_shape=(X.shape[1], X.shape[2]), activation='relu', return_sequences=False),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')  # Linear activation for regression
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=4, validation_data=(X_test, y_test), verbose=1)

# Evaluate on test set
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)

# Make Predictions
predictions = model.predict(X_test)
predicted_values = scaler_y.inverse_transform(predictions)
true_values = scaler_y.inverse_transform(y_test)

# Display evaluation results
print("Test Loss (MSE):", test_loss)
print("Test MAE:", test_mae)

# -------------------------------------------------------------
# 1. DATA VISUALIZATION
# -------------------------------------------------------------

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data_cleaned[numeric_columns].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Distribution of the target variable
plt.figure(figsize=(8, 6))
sns.histplot(data_cleaned['Savings for Property (£)'], kde=True, bins=30, color='skyblue')
plt.title("Distribution of Savings for Property (£)")
plt.xlabel("Savings for Property (£)")
plt.ylabel("Frequency")
plt.show()

# Predicted vs True Values Plot
plt.figure(figsize=(10, 6))
plt.plot(true_values[:50], label='True Values', marker='o')  # Limit to 50 for clarity
plt.plot(predicted_values[:50], label='Predicted Values', marker='x')
plt.title("True vs Predicted Savings for Property (£)")
plt.xlabel("Sample Index")
plt.ylabel("Savings (£)")
plt.legend()
plt.show()

# -------------------------------------------------------------
# 2. DECISION-MAKING SUPPORT
# -------------------------------------------------------------

# User-defined goal and interval
print("\n--- Decision-Making Support ---")
goal_savings = float(input("Enter your target savings (£): "))
time_interval = input("Enter time interval (daily/weekly/monthly): ").lower()

# Predicted average savings
predicted_average = predicted_values.mean()
print(f"Predicted Average Savings per month: £{predicted_average:.2f}")

# Recommendations based on goal and interval
if time_interval == 'daily':
    interval_factor = 30  # Approx. days in a month
elif time_interval == 'weekly':
    interval_factor = 4  # Approx. weeks in a month
else:  # Default to monthly
    interval_factor = 1

savings_per_interval = predicted_average / interval_factor
required_savings_per_interval = goal_savings / interval_factor

print(f"Predicted Savings per {time_interval}: £{savings_per_interval:.2f}")
print(f"Required Savings per {time_interval} to meet the goal: £{required_savings_per_interval:.2f}")

if savings_per_interval >= required_savings_per_interval:
    print("Recommendation: You are on track to meet your goal!")
else:
    print(f"Recommendation: Increase savings by £{required_savings_per_interval - savings_per_interval:.2f} per {time_interval} to meet your goal.")
