import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Download Data
data = yf.download("AAPL", start="2020-01-01", end="2024-01-01")
data.dropna(inplace=True)

# Step 2: Create Target Variable (1 if next day's close is higher)
data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
data = data[:-1]  # Drop last row with NaN target

# Step 3: Feature Selection
features = data[["Open", "High", "Low", "Close", "Volume"]]
target = data["Target"]

# Step 4: Time-Based Train/Test Split (70% train, 30% test)
split_index = int(len(data) * 0.7)
X_train, X_test = features[:split_index], features[split_index:]
y_train, y_test = target[:split_index], target[split_index:]

# Step 5: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Model Training
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)  # y_train is 1D Series

# Step 7: Prediction
predictions = model.predict(X_test_scaled)

# Step 8: Evaluation
print("Accuracy:", accuracy_score(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))

# Step 9: Plot Predicted vs Actual (First 50 test points)
plt.figure(figsize=(10, 5))
plt.plot(y_test.iloc[:50].values, label='Actual', marker='o', linestyle='--')
plt.plot(predictions[:50], label='Predicted', marker='x', linestyle='-')
plt.title('Actual vs Predicted Stock Movement (First 50 Days of Test Set)')
plt.xlabel('Day')
plt.ylabel('Direction (1 = Up, 0 = Down)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
