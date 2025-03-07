# 🚀 **Pure Transformer for Time-Series Classification (Without CNN)**  

Since you're using **Transformers for the first time**, I'll break down every component in detail, explaining **what each function does and how it affects the model**.  

---

## **📌 Why Use a Pure Transformer?**
✅ **Captures long-range dependencies** better than LSTMs.  
✅ **No need for CNN feature extraction**—relies solely on **self-attention**.  
✅ **Processes the entire sequence in parallel**, unlike LSTMs, which are sequential.  
✅ **Great for large datasets** where spatial features aren't necessary.  

---

## **🔹 Transformer Model for Time-Series**
Here’s a **pure Transformer** model **without CNN** for your dataset.

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

# Standardize features
scaler = StandardScaler()
data_reshaped = scaler.fit_transform(data.reshape(-1, 8)).reshape(402, 16000, 8)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_reshaped, labels, test_size=0.2, stratify=labels, random_state=42)

# Define Model Hyperparameters
time_steps = X_train.shape[1]  # 16000 time steps
features = X_train.shape[2]  # 8 features (sensor channels)
output_dim = len(np.unique(labels))  # Number of classes

# -------------------- Transformer Block --------------------
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    """Creates a Transformer Encoder Block"""
    
    # Multi-Head Self Attention: Learns relationships between time steps
    attention_output = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads)(inputs, inputs)
    attention_output = layers.Dropout(dropout)(attention_output)
    attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + inputs)  # Residual Connection

    # Feedforward Layer: Processes attention output further
    feedforward = layers.Dense(ff_dim, activation="relu")(attention_output)
    feedforward = layers.Dropout(dropout)(feedforward)
    feedforward = layers.Dense(inputs.shape[-1])(feedforward)
    output = layers.LayerNormalization(epsilon=1e-6)(feedforward + attention_output)  # Residual Connection
    
    return output

# -------------------- Build Pure Transformer Model --------------------
inputs = layers.Input(shape=(time_steps, features))

# Positional Encoding (Optional for improving Transformer performance)
positional_encoding = layers.Dense(features, activation="relu")(inputs)

# Stack multiple Transformer blocks
x = transformer_encoder(positional_encoding, head_size=64, num_heads=4, ff_dim=128)
x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128)

# Flatten and Fully Connected Layers
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(output_dim, activation='softmax')(x)  # Classification Output

# Compile Model
model_transformer = models.Model(inputs, outputs)
optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4, clipnorm=1.0)
model_transformer.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Show Model Summary
model_transformer.summary()

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

# Train Model
model_transformer.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test),
                    callbacks=[early_stop, lr_reducer])
```

---

# 📖 **Editorial & Deep Explanation (For Beginners)**
Since this is your **first time using Transformers**, let’s go step by step.  

---

## **🔹 Transformer Block Breakdown**
```python
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
```
🔹 **Inputs**: The time-series data `(batch, time_steps, features)`.  
🔹 **head_size**: The dimension of each attention head.  
🔹 **num_heads**: Number of parallel attention layers.  
🔹 **ff_dim**: Size of the hidden layer in the feedforward network.  
🔹 **dropout**: Prevents overfitting.  

### ✅ **1. Multi-Head Self Attention**
```python
attention_output = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads)(inputs, inputs)
```
- **Self-attention** computes the relationships between different time steps.  
- **Multiple heads (`num_heads=4`)** capture diverse patterns.  
- **More heads = better feature representation**, but **higher memory cost**.

🔴 **Modifications**:  
- Increase `num_heads` for **better performance** (try `num_heads=8` for large data).  
- Reduce it for **faster training** on small datasets.  

---

### ✅ **2. Residual Connection & Layer Normalization**
```python
attention_output = layers.Dropout(dropout)(attention_output)
attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + inputs)
```
- **Residual Connection** ensures **gradients flow smoothly**, preventing vanishing gradients.  
- **Layer Normalization** stabilizes training.  

🔴 **Modifications**:  
- Remove **dropout** if the model is underfitting.  
- Reduce **epsilon (e.g., `1e-5`)** for fine-tuned learning.  

---

### ✅ **3. Feedforward Network**
```python
feedforward = layers.Dense(ff_dim, activation="relu")(attention_output)
feedforward = layers.Dropout(dropout)(feedforward)
feedforward = layers.Dense(inputs.shape[-1])(feedforward)
```
- **FFN (Feedforward Network)** processes the output of self-attention.  
- Uses **ReLU activation** for non-linearity.  

🔴 **Modifications**:  
- Increase `ff_dim` for **better learning capacity** (`ff_dim=256` or `ff_dim=512` for large data).  
- Reduce dropout for **faster training**.  

---

## **🔹 Full Model Breakdown**
```python
inputs = layers.Input(shape=(time_steps, features))
```
- **Defines input shape** as `(time_steps=16000, features=8)`.  

```python
positional_encoding = layers.Dense(features, activation="relu")(inputs)
```
- **Adds positional encoding** since Transformers **don’t understand order** like LSTMs.  

```python
x = transformer_encoder(positional_encoding, head_size=64, num_heads=4, ff_dim=128)
x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128)
```
- **Stacks two Transformer blocks** to learn deeper relationships.  

```python
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(output_dim, activation='softmax')(x)
```
- **Global pooling** reduces model size.  
- **Dense layers** map Transformer outputs to final class labels.  

---

## **📌 Summary Table**
| **Feature** | **CNN-Transformer** | **Pure Transformer** |
|------------|------------------|----------------|
| **Feature Extraction** | CNN | None |
| **Temporal Modeling** | Transformer | Transformer |
| **Speed** | Faster | Slower |
| **Long Dependencies** | Good | Excellent |
| **Data Requirement** | Moderate | Large Datasets Needed |

---

## **🚀 When to Use This Pure Transformer Model?**
✅ **Large Datasets** (Transformers perform best when you have a LOT of data).  
✅ **Time-series data where order matters** (motion tracking, financial trends, EEG signals).  
✅ **Parallel Training Needed** (LSTMs are slow, Transformers can train on GPUs much faster).  

Would you like **a hybrid Transformer-RNN model** or **Hyperparameter tuning tips** next? 🚀🔥
