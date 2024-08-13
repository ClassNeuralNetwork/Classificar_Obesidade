import numpy as np
import pandas as pd
import matplotlib as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

input_train = pd.read_csv('/home/alanzin/Desktop/Facul 2024.1/Classificar_Obesidade/dataset/train/input_train.csv')
output_train = pd.read_csv('/home/alanzin/Desktop/Facul 2024.1/Classificar_Obesidade/dataset/train/output_train.csv')
input_test = pd.read_csv('/home/alanzin/Desktop/Facul 2024.1/Classificar_Obesidade/dataset/test/input_test.csv')
output_test = pd.read_csv('/home/alanzin/Desktop/Facul 2024.1/Classificar_Obesidade/dataset/test//output_test.csv')

# Normalizar os dados
scaler_input = StandardScaler()
input_train = scaler_input.fit_transform(input_train)
input_test = scaler_input.transform(input_test)

pd.DataFrame(input_train).to_csv('/home/alanzin/Desktop/Facul 2024.1/Classificar_Obesidade/model/input_train_standard.csv', index=False)
pd.DataFrame(input_test).to_csv('/home/alanzin/Desktop/Facul 2024.1/Classificar_Obesidade/model/input_test_standard.csv', index=False)

# Definir o modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(input_train.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='linear')  # Saída única para regressão
])

# Configurar early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)

# Compilar o modelo
model.compile(optimizer='adam', loss='mse', metrics=['mse'])

# Resumir o modelo
model.summary()

# Treinar o modelo
history = model.fit(input_train, output_train, epochs=200, validation_split=0.2, callbacks=[early_stopping])

pd.DataFrame(history.history).to_csv('/home/alanzin/Desktop/Facul 2024.1/Classificar_Obesidade/model/loss.csv', index=False)

model.save('/home/alanzin/Desktop/Facul 2024.1/Classificar_Obesidade/model/model.keras')