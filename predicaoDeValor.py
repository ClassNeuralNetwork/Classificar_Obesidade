import numpy as np
import tensorflow as tf
from tensorflow import keras

'''
Problem:
Y = 2 * X - 1
'''

def main():
  
    model = keras.Sequential([keras.layers.Dense(units=1, activation = 'linear', input_shape=[1])])
    model.compile('sgd', 'mean_squared_error')

    xs = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    ys = np.array([100000, 150000, 200000, 250000, 300000, 350000], dtype=float)

    model.fit(xs, ys, epochs=3000)

    x_new = np.array([7.0, 1.0])
    print(np.ceil(model.predict(x_new)))
    print(model.predict(x_new))
    model.summary()


if __name__ == '__main__':
    main()