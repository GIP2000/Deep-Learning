
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
from dataclasses import dataclass, field
from absl import flags,app
from tqdm import trange

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

class Dense(tf.Module):
    def __init__(self, neurons: int, is_output:bool = False, name: str = None):
        super().__init__(name=name)
        self.neurons = neurons
        self.is_output= is_output
        self.__is_built = False

    def build(self,rng,inputs:int, points:int):
        self.w = tf.Variable(rng.normal(shape=[inputs,self.neurons]),name = "w")
        self.b = tf.Variable(rng.normal(shape=[points,self.neurons]),name = "b")
        self.__is_built = True

    def __call__(self,x):
        if not self.__is_built:
            self.build()
        v = x @ self.w + self.b
        return tf.nn.sigmoid(v) if self.is_output else tf.nn.relu(v)

class Model(tf.Module):
    def __init__(self,rng,inputs: int, points: int, nodes:list[Dense], name = None):
        super().__init__(name=name)
        self.layers = []
        with self.name_scope:
            for node in nodes:
                node.build(rng,inputs,points)
                self.layers.append(node)
                inputs = node.neurons
    
    def __call__(self,x):
        value = x
        for node in self.layers:
            value = node(value)
        return value

def loss(y, y_hat): 
    return -y * tf.math.log(y_hat) - (1-y) * tf.math.log(1-y_hat)


FLAGS = flags.FLAGS
flags.DEFINE_integer("num_points",100, "Number of points in each spiral")
flags.DEFINE_integer("batch_size", 10, "Number of samples in batch")
flags.DEFINE_integer("random_seed", 31415, "Random seed")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
flags.DEFINE_integer("num_iters", 300, "Number of SGD iterations")


def convert_to_color(val: float): 
    return "blue" if val <= .5 else "red"

def main(a):
    d = Data(FLAGS.num_points)
    seed_sequence = np.random.SeedSequence(FLAGS.random_seed)
    np_seed, tf_seed = seed_sequence.spawn(2)
    tf_rng = tf.random.Generator.from_seed(tf_seed.entropy)


    x_train,y_train,x_test,y_test= d.split_data(int(FLAGS.num_points * 2 * .8))

    model = Model(tf_rng,2,int(FLAGS.num_points * 2 * .8),[
        Dense(32),
        Dense(32), 
        Dense(32), 
        Dense(32), 
        Dense(1,True), 
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=.01)

    bar = trange(FLAGS.num_iters)
    for i in bar:
        with tf.GradientTape() as tape:
            y_hat = model(x_train)
            ls = loss(y_train,y_hat)

        grads = tape.gradient(ls, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # bar.set_description(f"Loss @ {i} => {np.average(ls).reduce:0.6f}")
        bar.refresh()

    print("hi", y_hat.shape)
    true_colors = [convert_to_color(y) for y in d.y]
    # predictions = [convert_to_color(model.predict(points.reshape(2,1)).flatten()[0]) for points in d.x]
    # fig, ax = plt.subplots(1, 2, figsize=(10, 3), dpi=200)
    # ax[0].set_title("true")
    # ax[0].scatter(d.x[::,0],d.x[::,1],  color=true_colors)
    # ax[1].set_title("predictions")
    # ax[1].scatter(d.x[::,0],d.x[::,1],  color=predictions)

    plt.tight_layout()
    plt.savefig(f"{script_path}/fit.pdf")

if __name__ == "__main__":
    app.run(main)
