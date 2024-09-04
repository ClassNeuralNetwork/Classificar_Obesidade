import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score

# Carregar os dados
input_train = pd.read_csv('/home/alanzin/Desktop/Facul 2024.1/Classificar_Obesidade/dataset/train/input_train.csv')
output_train = pd.read_csv('/home/alanzin/Desktop/Facul 2024.1/Classificar_Obesidade/dataset/train/output_train.csv')
input_test = pd.read_csv('/home/alanzin/Desktop/Facul 2024.1/Classificar_Obesidade/dataset/test/input_test.csv')
output_test = pd.read_csv('/home/alanzin/Desktop/Facul 2024.1/Classificar_Obesidade/dataset/test/output_test.csv')

# Normalizar os dados
scaler_input = StandardScaler()
input_train = scaler_input.fit_transform(input_train)
input_test = scaler_input.transform(input_test)

# Salvar os dados normalizados
pd.DataFrame(input_train).to_csv('/home/alanzin/Desktop/Facul 2024.1/Classificar_Obesidade/model/input_train_standard.csv', index=False)
pd.DataFrame(input_test).to_csv('/home/alanzin/Desktop/Facul 2024.1/Classificar_Obesidade/model/input_test_standard.csv', index=False)
    
# Definir o modelo de classificação multi-classe
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='sigmoid', input_shape=(input_train.shape[1],), kernel_regularizer=tf.keras.regularizers.L1L2(
    l1=0.0, l2=0.00
)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='sigmoid'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='sigmoid'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(7, activation='softmax')
])

# Configurar early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)

# Compilar o modelo
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Resumir o modelo
model.summary()

# Treinar o modelo
history = model.fit(input_train, output_train, epochs=300, validation_split=0.2, shuffle=True, callbacks=[early_stopping])

# Salvar o histórico de treinamento
pd.DataFrame(history.history).to_csv('/home/alanzin/Desktop/Facul 2024.1/Classificar_Obesidade/model/loss.csv', index=False)

# Salvar o modelo treinado
model.save('/home/alanzin/Desktop/Facul 2024.1/Classificar_Obesidade/model/model.keras')

# Gerar gráfico da função perda
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()
plt.yscale("log")
plt.grid(True)
plt.savefig('/home/alanzin/Desktop/Facul 2024.1/Classificar_Obesidade/plots/funcao_perda.png')
plt.close()

# Fazer previsões com o modelo treinado
output_pred = model.predict(input_test)
output_pred = np.argmax(output_pred, axis=1)  # Obter a classe com a maior probabilidade

# Calcular métricas de avaliação
conf_matrix = confusion_matrix(output_test, output_pred)
accuracy = accuracy_score(output_test, output_pred)
precision = precision_score(output_test, output_pred, average='weighted')
recall = recall_score(output_test, output_pred, average='weighted')
f1 = f1_score(output_test, output_pred, average='weighted')

# Exibir a matriz de confusão
ConfusionMatrixDisplay(conf_matrix).plot()
plt.savefig('/home/alanzin/Desktop/Facul 2024.1/Classificar_Obesidade/plots/matriz_confusao.eps', format='eps')

# Exibir as métricas
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")