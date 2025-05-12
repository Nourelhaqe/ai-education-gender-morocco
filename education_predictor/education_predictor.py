import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Step 1: Load and prepare data
df = pd.read_csv("API_SE.PRM.ENRL.FE.ZS_DS2_en_csv_v2_96054.csv", skiprows=4)
morocco_df = df[df['Country Name'] == 'Morocco']

# Clean and reshape
morocco_df = morocco_df.drop(columns=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'])
morocco_df = morocco_df.T.reset_index()
morocco_df.columns = ['Year', 'Percent_Female']
morocco_df = morocco_df.dropna()
morocco_df['Year'] = morocco_df['Year'].astype(int)
morocco_df['Percent_Female'] = pd.to_numeric(morocco_df['Percent_Female'], errors='coerce')
morocco_df = morocco_df.dropna()

# Step 2: Normalize data
X = morocco_df['Year'].values.reshape(-1, 1)
y = morocco_df['Percent_Female'].values.reshape(-1, 1)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Step 4: Build the neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Step 5: Train the model
history = model.fit(X_train, y_train, epochs=200, verbose=1, validation_split=0.1)

# Step 6: Predict and visualize
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_actual = scaler_y.inverse_transform(y_test)

X_test_actual = scaler_X.inverse_transform(X_test)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(X_test_actual, y_actual, label="Actual", color="blue")
plt.scatter(X_test_actual, y_pred, label="Predicted", color="red")
plt.title("Actual vs Predicted % of Female Pupils in Morocco")
plt.xlabel("Year")
plt.ylabel("Female Enrollment (%)")
plt.legend()
plt.grid(True)
plt.show()
# Step 7: Predict future values (2025–2035)
future_years = np.array(range(2019, 2036)).reshape(-1, 1)
future_years_scaled = scaler_X.transform(future_years)
future_pred_scaled = model.predict(future_years_scaled)
future_pred = scaler_y.inverse_transform(future_pred_scaled)

# Plot future predictions
plt.figure(figsize=(10, 6))
plt.plot(morocco_df['Year'], morocco_df['Percent_Female'], label='Historical', marker='o')
plt.plot(future_years, future_pred, label='Predicted Future', marker='x', color='green')
plt.title('Predicted % of Female Pupils in Morocco (2019–2035)')
plt.xlabel('Year')
plt.ylabel('% Female Enrollment')
plt.grid(True)
plt.legend()
plt.show()
