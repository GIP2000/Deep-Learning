
#!/bin/env python3.8

"""
Homework Assignment #2: Gregory Presser
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
from dataclasses import dataclass, field
from absl import flags,app

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
        TODO: Ask if I need to add like variation here
        """
        self.index = np.arange(self.num_samples)
        phi = self.index / 16 * np.pi
        r = 6.5 * ((104 - self.index)/104)
        spiral1 = np.array([r * np.cos(phi) / 13 + .5,r * np.sin(phi) / 13 + .5])
        spiral2 = np.array([-r * np.cos(phi) / 13 + 0.5,-r * np.sin(phi) / 13 + 0.5])

        self.x = np.concatenate((spiral1,spiral2),axis=1)
        self.y = np.concatenate((np.zeros(self.num_samples), np.ones(self.num_samples)),axis=None)

        p = np.random.permutation(len(self.y))
        self.x = self.x[::,p].transpose()
        self.y = self.y[p].transpose()



    def split_data(self, split_point:int):
        return (
                tf.convert_to_tensor(self.x[0:split_point,::],dtype=tf.float32),
                tf.convert_to_tensor(self.y[0:split_point].reshape((-1,1)),dtype=tf.float32),
                tf.convert_to_tensor(self.x[split_point:,::],dtype=tf.float32),
                tf.convert_to_tensor(self.y[split_point:].reshape((-1,1)),dtype= tf.float32)
               )


FLAGS = flags.FLAGS
flags.DEFINE_integer("num_points",100, "Number of points in each spiral")
flags.DEFINE_integer("batch_size", 10, "Number of samples in batch")


def convert_to_color(val: int): 
    return "blue" if val == 0 else "red"

def main(a):
    d = Data(FLAGS.num_points)

    x_train,y_train,x_test,y_test= d.split_data(int(FLAGS.num_points * 2 * .8))

    model = Sequential([
        Input(shape=(2,1)),
        Dense(4,activation='relu'),
        Dense(7,activation='relu'),
        Dense(4,activation='relu'),
        Dense(2,activation='relu'),
        Dense(1,activation='sigmoid'),
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=.01)

    model.compile(optimizer=optimizer,loss="binary_crossentropy",metrics=['accuracy'])

    model.fit(x_train,y_train, epochs=2000,batch_size=FLAGS.batch_size, validation_split=.2)

    results = model.evaluate(x_test,y_test,verbose = 0)
    print("results",results)


    true_colors = [convert_to_color(y) for y in d.y]
    predictions = [convert_to_color(np.argmin(model.predict(points.reshape(2,1)).flatten())) for points in d.x]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3), dpi=200)
    ax[0].set_title("true")
    ax[0].scatter(d.x[::,0],d.x[::,1],  color=true_colors)
    ax[1].set_title("predictions")
    ax[1].scatter(d.x[::,0],d.x[::,1],  color=predictions)

    plt.tight_layout()
    plt.savefig(f"{script_path}/fit.pdf")

if __name__ == "__main__":
    app.run(main)
