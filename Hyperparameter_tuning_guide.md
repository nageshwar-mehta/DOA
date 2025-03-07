# **📌 Hyperparameter Tuning Guide for Your ML Models** 🚀🔥  

Hyperparameter tuning is **crucial** for optimizing model performance. Below, I provide a **detailed guide** for tuning different models, covering **MLP, CNN, LSTM, CNN+LSTM, and Transformers**.

---

## **🛠️ General Strategy for Hyperparameter Tuning**  
Before we dive into specific models, here are some **best practices** for hyperparameter tuning:  

### **1️⃣ Grid Search vs. Random Search vs. Bayesian Optimization**  
| **Method** | **Pros** | **Cons** | **Recommended for** |
|------------|---------|---------|--------------------|
| **Grid Search** (systematic search over all parameter combinations) | Ensures finding the best combination | Very slow for large hyperparameter spaces | Small models (MLP, CNN) |
| **Random Search** (randomly selects parameters) | Faster, explores more parameter space | May miss optimal values | CNN, LSTM |
| **Bayesian Optimization (Optuna/HyperOpt)** | Uses probability to find the best parameters efficiently | More complex implementation | Transformers, CNN + Transformer |

✅ **Recommendation:**  
- Use **Random Search** for CNN/LSTM models.  
- Use **Bayesian Optimization** for Transformers or large models.

---

# **📌 Model-Specific Hyperparameter Tuning**  

## **1️⃣ Tuning MLP (Dense Neural Network)**
| **Hyperparameter** | **Recommended Range** | **Effect** |
|--------------------|---------------------|-----------|
| Number of Layers | 2 - 5 | More layers → better learning but risk of overfitting |
| Neurons per Layer | 64 - 512 | More neurons → better learning but increases computation |
| Activation Function | ReLU, LeakyReLU, Swish | ReLU is fast, Swish is better for deep networks |
| Dropout Rate | 0.1 - 0.4 | Prevents overfitting |
| Learning Rate | `1e-4` to `1e-2` | Lower for deeper networks |

### **Suggested Grid Search Example**
```python
from sklearn.model_selection import ParameterGrid

param_grid = {
    'layers': [[128, 64], [256, 128, 64]],
    'activation': ['relu', 'swish'],
    'dropout': [0.1, 0.2, 0.3],
    'learning_rate': [1e-3, 1e-4]
}

for params in ParameterGrid(param_grid):
    print(params)  # Train the model with these hyperparameters
```

---

## **2️⃣ Tuning CNN (Convolutional Neural Network)**
| **Hyperparameter** | **Recommended Range** | **Effect** |
|--------------------|---------------------|-----------|
| Number of Conv Layers | 2 - 5 | More layers capture more patterns but slow training |
| Filters per Layer | 32 - 256 | More filters → better feature extraction but expensive |
| Kernel Size | 3, 5, 7 | Larger kernels capture wider patterns |
| Pooling Type | MaxPooling, AveragePooling | MaxPooling retains high values, AveragePooling smoothens |
| Dropout | 0.2 - 0.5 | Prevents overfitting |
| Learning Rate | `1e-4` to `1e-2` | Lower for deeper networks |

✅ **Use Random Search for CNNs**:
```python
from keras_tuner import RandomSearch

def build_cnn(hp):
    model = tf.keras.Sequential([
        layers.Conv1D(filters=hp.Choice('filters', [64, 128, 256]), 
                      kernel_size=hp.Choice('kernel_size', [3, 5, 7]),
                      activation='relu', input_shape=(8, 1)),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(hp.Float('dropout', 0.2, 0.5, step=0.1)),
        layers.Dense(4, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Choice('lr', [1e-2, 1e-3, 1e-4])),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

tuner = RandomSearch(build_cnn, objective='val_accuracy', max_trials=10)
tuner.search(X_train_reshaped, y_train, epochs=10, validation_split=0.2)
```

---

## **3️⃣ Tuning LSTM (Long Short-Term Memory)**
| **Hyperparameter** | **Recommended Range** | **Effect** |
|--------------------|---------------------|-----------|
| Number of LSTM Units | 64 - 512 | More units → better memory but slower training |
| Number of Layers | 1 - 3 | More layers → better learning but risk of overfitting |
| Dropout Rate | 0.1 - 0.5 | Prevents overfitting |
| Learning Rate | `1e-4` to `1e-2` | Lower for deeper networks |

✅ **Use Bayesian Optimization (Optuna) for LSTMs**
```python
import optuna

def objective(trial):
    lstm_units = trial.suggest_int('lstm_units', 64, 512, step=64)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_loguniform('lr', 1e-4, 1e-2)

    model = tf.keras.Sequential([
        layers.LSTM(lstm_units, return_sequences=True, input_shape=(8, 1)),
        layers.LSTM(lstm_units // 2),
        layers.Dropout(dropout),
        layers.Dense(4, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=0)
    return max(history.history['val_accuracy'])

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)
print(study.best_params)
```

---

## **4️⃣ Tuning Transformer Models**
| **Hyperparameter** | **Recommended Range** | **Effect** |
|--------------------|---------------------|-----------|
| Number of Heads | 4 - 8 | More heads → better parallel attention |
| Depth of Transformer Blocks | 2 - 6 | More depth captures better relations but is slower |
| Feedforward Neurons | 128 - 512 | More neurons → better representation |
| Learning Rate | `1e-5` to `1e-3` | Lower learning rate prevents overfitting |

✅ **Use Keras Tuner for Transformer**
```python
from keras_tuner import BayesianOptimization

def build_transformer(hp):
    input_layer = layers.Input(shape=(8, 1))
    x = layers.MultiHeadAttention(num_heads=hp.Choice('heads', [4, 8]), key_dim=64)(input_layer, input_layer)
    x = layers.Dense(hp.Int('dense_units', 128, 512, step=128), activation='relu')(x)
    x = layers.Dense(4, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=x)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Choice('lr', [1e-4, 1e-5])),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

tuner = BayesianOptimization(build_transformer, objective='val_accuracy', max_trials=10)
tuner.search(X_train, y_train, epochs=10, validation_split=0.2)
```

---

# **📌 Summary**
| **Model** | **Tuning Method** | **Key Hyperparameters** |
|----------|----------------|----------------------|
| **MLP (Dense)** | Grid Search | Layers, neurons, dropout, learning rate |
| **CNN** | Random Search | Filters, kernel size, pooling, dropout |
| **LSTM** | Bayesian Optimization | LSTM units, layers, dropout, learning rate |
| **Transformer** | Bayesian Optimization | Attention heads, feedforward neurons, learning rate |

---

# **🚀 Final Recommendation**
- **For quick tuning:** Use **Random Search** for CNNs.  
- **For best accuracy:** Use **Bayesian Optimization (Optuna/Keras Tuner)**.  
- **For Transformer-based models:** Keep **learning rate very low (1e-5 to 1e-4)** to prevent overfitting.

Let me know if you need **hyperparameter tuning scripts for specific datasets**! 🚀🔥
