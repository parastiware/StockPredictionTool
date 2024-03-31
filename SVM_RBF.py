import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# Load the data from CSV file
data = pd.read_csv('portfolio_data.csv')

# Convert 'Date' column to numeric format (number of days since a reference date)
data['Date'] = (pd.to_datetime(data['Date']) - pd.to_datetime('1970-01-01')).dt.days

# Extract features (Date) and target variable (AMZN)
X = data['Date'].values.reshape(-1, 1)
y = data['AMZN'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the SVM model with RBF kernel
svm_rbf_model = SVR(kernel='rbf')
svm_rbf_model.fit(X_train, y_train)

# Making predictions on the testing set
svm_rbf_predictions = svm_rbf_model.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, svm_rbf_predictions)
print("Mean Squared Error:", mse)
reference_date = pd.to_datetime('1970-01-01')
actual_dates = [reference_date + pd.DateOffset(days=int(date)) for date in X_test.flatten()]

# Plotting the original and predicted data
plt.figure(figsize=(10, 6))
plt.scatter(actual_dates, y_test, color='blue', label='Original Data')
plt.scatter(actual_dates, svm_rbf_predictions, color='red', label='Predicted Data')
plt.title('Original vs Predicted Amazon Stock (SVM with RBF Kernel)')
plt.xlabel('Date')
plt.ylabel('AMZN Stock Price')
plt.legend()
plt.grid(True)
plt.show()
