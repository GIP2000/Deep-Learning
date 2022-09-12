
#!/bin/env python3.8

"""
Homework Assignment #1: Gregory Presser
"""
import os
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from dataclasses import dataclass, field, InitVar
from absl import flags,app
import itertools

script_path = os.path.dirname(os.path.realpath(__file__))


@dataclass
class Data:
    num_samples: int
    x: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)

    def __post_init__(self): 
        """
        Data Generation based off of https://conx.readthedocs.io/en/latest/Two-Spirals.html
        Then Vectorized
        """
        self.index = np.arange(self.num_samples)
        phi = self.index / 16 * np.pi
        r = 6.5 * ((104 - self.index)/104)
        spiral1 = np.array([r * np.cos(phi) / 13 + .5,r * np.sin(phi) / 13 + .5])
        spiral2 = np.array([-r * np.cos(phi) / 13 + 0.5,-r * np.sin(phi) / 13 + 0.5])

        self.x = np.concatenate((spiral1,spiral2),axis=1).transpose()
        self.y = np.concatenate((np.zeros(self.num_samples), np.ones(self.num_samples)),axis=None).transpose()

        # p = np.random.permutation(len(self.y))
        # self.x = self.x[::,p].transpose()
        # self.y = self.y[p].transpose()



    def split_data(self, split_point):
        return (tf.convert_to_tensor(self.x[0:split_point,::],dtype=tf.float32),tf.convert_to_tensor(self.y[0:split_point].reshape((-1,1)),dtype=tf.float32),tf.convert_to_tensor(self.x[split_point:,::],dtype=tf.float32),tf.convert_to_tensor(self.y[split_point:].reshape((-1,1)),dtype= tf.float32))


FLAGS = flags.FLAGS
flags.DEFINE_integer("num_points",100, "Number of points in each spiral")
flags.DEFINE_integer("batch_size", 16, "Number of samples in batch")

def main(a):
    d = Data(FLAGS.num_points)

    x_train,y_train,x_test,y_test= d.split_data(int(FLAGS.num_points * 2 * .8))
    print(x_train.shape)

    model = Sequential([
        Input(shape=(2,1)),
        Dense(5,activation='sigmoid'),
        Dense(5,activation='sigmoid'),
        Dense(5,activation='sigmoid'),
        Dense(1,activation='sigmoid'),
    ])

    model.compile(optimizer='adam',loss="binary_crossentropy",metrics=['accuracy'])

    model.fit(x_train,y_train, epochs=100,batch_size=FLAGS.batch_size, validation_split=.2)

    results = model.evaluate(x_test,y_test,verbose = 0)
    print("results",results)


    predictions = ["blue" if np.argmin(model.predict(points.reshape(2,1)).flatten()) == 0 else "red" for points in d.x]
    plt.scatter(d.x[::,0],d.x[::,1],  color=predictions)



    plt.tight_layout()
    plt.savefig(f"{script_path}/fit.pdf")

if __name__ == "__main__":
    app.run(main)
