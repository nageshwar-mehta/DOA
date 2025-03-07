import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import optuna
from keras_tuner import RandomSearch, BayesianOptimization
from tensorflow.keras import layers

# Load Data
data = np.load('/kaggle/input/time-domain/time_domain_data.npy')  # Shape: (402, 16000, 8)
labels = np.load('/kaggle/input/time-domain/labels.npy')  # Shape: (402,)

# Normalize Data
data /= np.max(np.abs(data))  # Scale between -1 and 1

# Reshape data (Flatten time-series into features per sensor)
data_2d = data.reshape(-1, 8)  # Shape: (402 * 16000, 8)
labels_2d = np.repeat(labels, 16000)  # Shape: (6432000,)

# Standardize features
scaler = StandardScaler()
data_2d = scaler.fit_transform(data_2d)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(data_2d, labels_2d, test_size=0.2, stratify=labels_2d, random_state=42)

### MLP MODEL WITH OPTUNA ###
def objective(trial):
    model = tf.keras.Sequential([
        layers.Dense(trial.suggest_int('units1', 64, 512, step=64), activation='relu', input_shape=(8,)),
        layers.Dropout(trial.suggest_float('dropout1', 0.1, 0.5)),
        layers.Dense(trial.suggest_int('units2', 64, 256, step=64), activation='relu'),
        layers.Dropout(trial.suggest_float('dropout2', 0.1, 0.5)),
        layers.Dense(len(np.unique(labels)), activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=trial.suggest_loguniform('lr', 1e-4, 1e-2)),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=0)
    return max(history.history['val_accuracy'])

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)
print("Best parameters:", study.best_params)

### CNN MODEL WITH KERAS TUNER ###
def build_cnn(hp):
    model = tf.keras.Sequential([
        layers.Conv1D(filters=hp.Choice('filters', [64, 128, 256]), kernel_size=hp.Choice('kernel_size', [3, 5]),
                      activation='relu', input_shape=(8, 1)),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(hp.Float('dropout', 0.2, 0.5, step=0.1)),
        layers.Dense(len(np.unique(labels)), activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Choice('lr', [1e-2, 1e-3, 1e-4])),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

tuner = RandomSearch(build_cnn, objective='val_accuracy', max_trials=10)
tuner.search(X_train.reshape(-1, 8, 1), y_train, epochs=10, validation_split=0.2)
print("Best CNN parameters:", tuner.get_best_hyperparameters()[0].values)

### LSTM MODEL WITH OPTUNA ###
def lstm_objective(trial):
    model = tf.keras.Sequential([
        layers.LSTM(trial.suggest_int('lstm_units', 64, 512, step=64), return_sequences=True, input_shape=(8, 1)),
        layers.LSTM(trial.suggest_int('lstm_units2', 32, 256, step=32)),
        layers.Dropout(trial.suggest_float('dropout', 0.1, 0.5)),
        layers.Dense(len(np.unique(labels)), activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=trial.suggest_loguniform('lr', 1e-4, 1e-2)),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train.reshape(-1, 8, 1), y_train, epochs=10, validation_data=(X_test.reshape(-1, 8, 1), y_test), verbose=0)
    return max(history.history['val_accuracy'])

study = optuna.create_study(direction='maximize')
study.optimize(lstm_objective, n_trials=10)
print("Best LSTM parameters:", study.best_params)

### TRANSFORMER MODEL WITH BAYESIAN OPTIMIZATION ###
def build_transformer(hp):
    input_layer = layers.Input(shape=(8, 1))
    x = layers.MultiHeadAttention(num_heads=hp.Choice('heads', [4, 8]), key_dim=64)(input_layer, input_layer)
    x = layers.Dense(hp.Int('dense_units', 128, 512, step=128), activation='relu')(x)
    x = layers.Dense(len(np.unique(labels)), activation='softmax')(x)
    model = tf.keras.Model(inputs=input_layer, outputs=x)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Choice('lr', [1e-4, 1e-5])),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

tuner = BayesianOptimization(build_transformer, objective='val_accuracy', max_trials=10)
tuner.search(X_train.reshape(-1, 8, 1), y_train, epochs=10, validation_split=0.2)
print("Best Transformer parameters:", tuner.get_best_hyperparameters()[0].values)
