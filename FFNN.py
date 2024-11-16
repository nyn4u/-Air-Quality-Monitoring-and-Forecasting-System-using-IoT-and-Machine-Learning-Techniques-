from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load and preprocess your data
data = pd.read_csv("air_quality_data.csv")  # Replace with your data file
X = data[['feature1', 'feature2', 'feature3']].values  # Replace with your features
y = data['pollutant_value'].values  # Replace 'pollutant_value' with your target column

# Normalize data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1, 1))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build FFNN model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # Output layer
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# Predict
predicted = model.predict(X_test)
predicted = scaler_y.inverse_transform(predicted)
y_test = scaler_y.inverse_transform(y_test)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual')
plt.plot(predicted, label='Predicted')
plt.xlabel('Time')
plt.ylabel('Pollutant Value')
plt.title('FFNN Model Prediction')
plt.legend()
plt.show()

# Evaluate the model
mae = mean_absolute_error(y_test, predicted)
mse = mean_squared_error(y_test, predicted)
print(f'Mean Absolute Error: {mae}, Mean Squared Error: {mse}')
