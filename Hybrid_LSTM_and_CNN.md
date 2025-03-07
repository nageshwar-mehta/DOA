Here’s a complete **Hybrid CNN-LSTM model** optimized for your sensor data classification problem, along with a **detailed editorial** explaining every part of the code. This will help you understand **what the model does, how to modify it**, and how changes impact performance.

---

# **📌 Why Use a Hybrid CNN-LSTM Model?**
A **CNN extracts spatial features** from short time sequences, while an **LSTM captures long-term dependencies** in time-series data. Combining them allows:
- **CNN** to recognize **local sensor patterns** (short-term correlations).
- **LSTM** to **retain sequential memory** (long-term dependencies).

---

## **🚀 Optimized CNN-LSTM Model Code**
```python
# Import required libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Data
data = np.load('/kaggle/input/time-domain/time_domain_data.npy')  # Shape: (402, 16000, 8)
labels = np.load('/kaggle/input/time-domain/labels.npy')  # Shape: (402,)

# Normalize Data
data /= np.max(np.abs(data))  # Scale between -1 and 1

# Reshape data for CNN-LSTM
data_reshaped = data.reshape(402, 16000, 8)  # Shape: (samples, time steps, features)

# Standardize features
scaler = StandardScaler()
data_reshaped = scaler.fit_transform(data_reshaped.reshape(-1, 8)).reshape(402, 16000, 8)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_reshaped, labels, test_size=0.2, stratify=labels, random_state=42)

# Define Model Hyperparameters
time_steps = X_train.shape[1]  # 16000 time steps
features = X_train.shape[2]  # 8 features (sensor channels)
output_dim = len(np.unique(labels))  # Number of classes

# Build Hybrid CNN-LSTM Model
model_cnn_lstm = models.Sequential([
    # Convolutional Layers (Feature Extraction)
    layers.Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(time_steps, features)),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    
    # LSTM Layer (Temporal Dependencies)
    layers.LSTM(64, return_sequences=True),
    layers.LSTM(32, return_sequences=False),
    
    # Fully Connected Dense Layers
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(output_dim, activation='softmax')
])

# Compile Model
optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4, clipnorm=1.0)
model_cnn_lstm.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Show Model Summary
model_cnn_lstm.summary()

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

# Train Model
model_cnn_lstm.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test),
                    callbacks=[early_stop, lr_reducer])
```
---

# **📖 Editorial & Explanation of the Code**
### **🔹 1. Data Preparation**
- **Scaling the data** to the range [-1,1] ensures stable model training.
- **Standardizing features** helps improve learning and reduces bias.
- **Reshaping** to `(samples, time_steps, features)` is **essential** because:
  - **CNN needs the time series in 3D format.**
  - **LSTM expects sequential input.**

---

### **🔹 2. Model Architecture Breakdown**
#### **✅ Convolutional Feature Extractor (CNN)**
```python
layers.Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(time_steps, features)),
layers.MaxPooling1D(pool_size=2),
layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
layers.MaxPooling1D(pool_size=2),
```
- **First Conv1D Layer** (64 filters, kernel size = 5)
  - Captures **short-term sensor variations**.
- **MaxPooling1D**
  - Reduces dimensionality and extracts **important features**.
- **Second Conv1D Layer** (128 filters, kernel size = 3)
  - Extracts deeper **spatial patterns** from sensor data.

🔴 **Changing Kernel Size**:
- **Larger (e.g., 7)** → Captures broader dependencies but loses small details.
- **Smaller (e.g., 3)** → Focuses on finer details.

---

#### **✅ Long-Term Memory Learning (LSTM)**
```python
layers.LSTM(64, return_sequences=True),
layers.LSTM(32, return_sequences=False),
```
- **First LSTM (64 units, `return_sequences=True`)**
  - Keeps temporal relationships **for next LSTM layer**.
- **Second LSTM (32 units, `return_sequences=False`)**
  - Outputs a **final sequence representation** for classification.

🔴 **Changing LSTM Units**:
- **Higher (128+)** → Captures **longer-term dependencies** but increases memory.
- **Lower (32 or less)** → Less memory, but might miss some sequential patterns.

---

#### **✅ Fully Connected Layers**
```python
layers.Dense(64, activation='relu'),
layers.Dropout(0.2),
layers.Dense(output_dim, activation='softmax')
```
- **Dense(64)** → Helps process CNN-LSTM features before classification.
- **Dropout (0.2)** → Reduces overfitting.
- **Softmax Layer** → Final **classification output**.

🔴 **Changing Dropout**:
- **Increase (0.5)** if **overfitting** is an issue.
- **Decrease (0.1)** if the model is **not learning well**.

---

### **🔹 3. Model Compilation & Training**
```python
optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4, clipnorm=1.0)
model_cnn_lstm.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```
- **AdamW Optimizer** is better for large datasets.
- **Weight Decay** prevents overfitting.
- **Learning Rate (1e-3)** is **moderate**; if the model is learning too fast, reduce it.

🔴 **Changing Learning Rate**:
- **Lower (1e-4)** for slow but stable learning.
- **Higher (5e-3)** if the model is too slow.

---

### **🔹 4. Callbacks for Efficient Training**
```python
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
```
- **Early Stopping** stops training **if no improvement** in validation loss.
- **Learning Rate Reduction** helps improve accuracy **if the model plateaus**.

🔴 **Modifying Patience**:
- **Increase (patience=10)** if the model needs more time to learn.
- **Decrease (patience=3)** for **quicker training**.

---

## **📌 Summary Table**
| **Component**        | **Purpose** | **Impact of Changing** |
|---------------------|------------|------------------|
| **Conv1D Layers**  | Feature extraction | Larger kernel captures broader patterns |
| **MaxPooling1D**   | Reduces dimensionality | Too large loses important info |
| **LSTM Layers**    | Captures long-term dependencies | More units = better memory, but higher cost |
| **Dense Layers**   | Final classification | More units = better accuracy, but overfitting risk |
| **Dropout**        | Prevents overfitting | Too high = model learns slowly |
| **Learning Rate**  | Controls model speed | Too high = unstable, too low = slow |

---

# **🚀 Final Takeaways**
- **This Hybrid CNN-LSTM balances spatial & temporal learning.**
- **Adjust kernel sizes, LSTM units, and dropout based on performance.**
- **Use EarlyStopping & ReduceLROnPlateau to fine-tune automatically.**

Let me know if you need **further optimizations**! 🚀🔥
