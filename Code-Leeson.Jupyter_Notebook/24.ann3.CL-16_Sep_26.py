import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import time  # <--- Import the time module

# ----- data -----
X = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9]], dtype=float)
y = np.array([60,65,70,75,80,85,90,92,95], dtype=float).reshape(-1,1)

# ----- scale X and y -----
x_scaler = StandardScaler()
y_scaler = StandardScaler()
X_s = x_scaler.fit_transform(X)
y_s = y_scaler.fit_transform(y)

# ----- ANN: linear regression as a tiny net -----
tf.random.set_seed(42)
model = Sequential([Input(shape=(1,)), Dense(1, activation='linear')])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05), loss='mse')

# Start the timer
start_time = time.time()

model.fit(X_s, y_s, epochs=800, verbose=0)

# Stop the timer and calculate duration
end_time = time.time()
training_time = end_time - start_time

# predictions on train (sanity check)
pred_s = model.predict(X_s, verbose=0)
pred = y_scaler.inverse_transform(pred_s)
for h, p in zip(X.ravel(), pred.ravel()):
    print(f"hours={int(h)} -> predicted score≈ {p:.1f}")

# Print training time
print(f"\nTraining completed in: {training_time:.4f} seconds")

# ----- hours needed for score=100 -----
target_s = y_scaler.transform([[100.0]])[0,0]   # scale target
w, b = model.layers[0].get_weights()
w = float(w[0,0]); b = float(b[0])
x_needed_s = (target_s - b) / w                 # hours (scaled)
hours_needed = float(x_scaler.inverse_transform([[x_needed_s]])[0,0])
print(f"Hours needed for score 100 ≈ {hours_needed:.2f}")
