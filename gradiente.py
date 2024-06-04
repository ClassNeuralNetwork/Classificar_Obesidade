import numpy as np
import matplotlib.pyplot as ply
import tensorflow as tf
from tensorflow import keras

def mse (w,x,y):
    y_pred = x* w
    e = y_pred - y
    E = np.sum(e**2)
    return E

x = np.array([1, 2, 3, 4, 5, 6], dtype=float)
y = np.array([0, 1, 2, 3, 4, 5], dtype=float)

#Hyperparameters
learning_rate = 0.05
num_iterations = 30
num_samples = len(x)
w = 0
errors=[]
ypred = []

for i in range(num_iterations):
    y_pred = w * x
    e = y_pred - y
    dw = (2/num_samples) * np.sum(e*x)
    w = w-learning_rate * dw
    error = mse(w,x,y)
    errors.append(error)
    ypred.append(y_pred)
    print(f'Iteração : {i} ------- MSE {error}')

ply.plot(range(num_iterations), errors)
ply.xlabel('Iterações')
ply.ylabel('E(w,x,y)')
ply.title('Função Erro')
ply.show()