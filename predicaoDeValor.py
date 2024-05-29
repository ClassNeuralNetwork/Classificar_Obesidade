import numpy as np
import tensorflow as tf
from tensorflow import keras

'''
Problem:
Y = 2 * X - 1
'''

def main():
    # cria o modelo sequencial com 1 camada (1 neurônio) e 1 dado de entrada
    model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])]) #Dense representa uma camada de neurônios conectados
    model.compile(optimizer='sgd', loss='mean_squared_error') #define a forma como vai ser calculado o erro (médio quadrático)

    # inicializa os dados de entrada como dois arrays
    xs = np.array([1, 2, 3], dtype=float)
    ys = np.array([100000, 150000, 200000], dtype=float)

    # define o modelo (exemplos e épocas) para o treinamento
    model.fit(xs, ys, epochs=1000)

    # faz a predição do modelo para o valor de entrada
    x_new = np.array([7.0])
    print(np.ceil(model.predict(x_new)))
    print(model.predict(x_new))


if __name__ == '__main__':
    main()