import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import shap

# Carregar o modelo treinado
model = tf.keras.models.load_model('/home/alanzin/Desktop/Facul 2024.1/Classificar_Obesidade/model/model.keras')

# Features selecionadas
selected_feature = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'FAVC', 'NCP', 'CAEC', 'SMOKE']

# Carregar os dados de teste
input_test = pd.read_csv('/home/alanzin/Desktop/Facul 2024.1/Classificar_Obesidade/model/input_test_standard.csv', header=0, names=selected_feature)
input_train = pd.read_csv('/home/alanzin/Desktop/Facul 2024.1/Classificar_Obesidade/model/input_train_standard.csv', header=0, names=selected_feature)

# Resumir os dados de background com shap.kmeans
explainer = shap.KernelExplainer(model.predict, input_test)

# Gerar os shap_values para um subconjunto menor de input_test
shap_values = explainer.shap_values(input_test)

# Verificar a forma dos shap_values e input_test
print("Shape de shap_values:", np.array(shap_values).shape)
print("Shape de input_test_sample:", input_test.shape)

# Remover a dimensão extra dos shap_values (dimensão de tamanho 1)
shap_values = np.array(shap_values).squeeze(-1)

# Plots
plt.figure()
shap.summary_plot(shap_values, input_test, plot_type="bar", feature_names=input_test.columns)
plt.savefig('/home/alanzin/Desktop/Facul 2024.1/Classificar_Obesidade/plots/summary_plot_bar.eps', format='eps')

plt.figure()
shap.summary_plot(shap_values, input_test, feature_names=input_test.columns)
plt.savefig('/home/alanzin/Desktop/Facul 2024.1/Classificar_Obesidade/plots/summary_plot.eps', format='eps')

plt.figure()
shap.dependence_plot(selected_feature[0], shap_values, input_test)
plt.savefig('/home/alanzin/Desktop/Facul 2024.1/Classificar_Obesidade/plots/dependence_plot.eps', format='eps')