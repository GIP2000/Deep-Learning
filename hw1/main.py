#!/bin/env python3.8

"""
Homework Assignment #1: Gregory Presser
Help Recived From: Husam Almanakly 
"""
import os
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from absl import app
from absl import flags
from tqdm import trange

from dataclasses import dataclass, field, InitVar

script_path = os.path.dirname(os.path.realpath(__file__))


@dataclass
class LinearModel:
    weights: np.ndarray
    bias: float
    mew: np.ndarray
    sigma: np.ndarray


@dataclass
class Data:
    model: LinearModel
    rng: InitVar[np.random.Generator]
    num_features: int
    num_samples: int
    sigma: float
    x: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)

    def __post_init__(self, rng):
        self.index = np.arange(self.num_samples)
        self.x = rng.uniform(0.1, 0.9,
                             size=(self.num_samples, self.num_features))
        clean_y = np.sin(2*np.pi * self.x) 
        self.y = rng.normal(loc=clean_y, scale=self.sigma)

    def get_batch(self, rng, batch_size):
        """
        Select random subset of examples for training batch
        """
        choices = rng.choice(self.index, size=batch_size)

        return self.x[choices], self.y[choices].flatten()


def compare_linear_models(a: LinearModel, b: LinearModel):
    for w_a, w_b in zip(a.weights, b.weights):
        print(f"{w_a:0.2f}, {w_b:0.2f}")

    print(f"{a.bias:0.2f}, {b.bias:0.2f}")


font = {
    # "family": "Adobe Caslon Pro",
    "size": 10,
}

matplotlib.style.use("classic")
matplotlib.rc("font", **font)

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_features", 1, "Number of features in record")
flags.DEFINE_integer("num_samples", 50, "Number of samples in dataset")
flags.DEFINE_integer("batch_size", 16, "Number of samples in batch")
flags.DEFINE_integer("num_iters", 300, "Number of SGD iterations")
flags.DEFINE_float("learning_rate", 0.1, "Learning rate / step size for SGD")
flags.DEFINE_integer("random_seed", 31415, "Random seed")
flags.DEFINE_float("sigma_noise", 0.5,
                   "Standard deviation of noise random variable")
flags.DEFINE_bool("debug", False, "Set logging level to debug")


class Model(tf.Module):
    def __init__(self, rng, num_features):
        """
        A plain linear regression model with a bias term
        """
        self.num_features = num_features
        self.w = tf.Variable(rng.normal(shape=[self.num_features, 1]))
        self.mew = tf.Variable(tf.cast(tf.linspace(.1,.9, self.num_features),tf.float32))
        self.sigma = tf.Variable(rng.normal(shape=[self.num_features, 1]))
        self.b = tf.Variable(tf.ones(shape=[self.num_features, 1])) * .4

    def __call__(self, x):
        phi = tf.math.exp((-(x-self.mew)**2) / (self.sigma**2))
        return tf.squeeze((self.w * phi) + self.b)

    @property
    def model(self):
        return LinearModel(
            self.w.numpy().reshape([self.num_features]),
            self.b.numpy().squeeze(),
            self.mew.numpy().reshape([self.num_features]),
            self.sigma.numpy().reshape([self.num_features])
        )


def main(a):
    logging.basicConfig()

    if FLAGS.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Safe np and tf PRNG
    seed_sequence = np.random.SeedSequence(FLAGS.random_seed)
    np_seed, tf_seed = seed_sequence.spawn(2)
    np_rng = np.random.default_rng(np_seed)
    tf_rng = tf.random.Generator.from_seed(tf_seed.entropy)

    data_generating_model = LinearModel(
        weights=np_rng.integers(low=0, high=5, size=(FLAGS.num_features)),
        bias=2,
        mew=np_rng.integers(low=0, high=1, size=(FLAGS.num_features)),
        sigma=np_rng.integers(low=0, high=1, size=(FLAGS.num_features))
    )
    logging.debug(data_generating_model)

    data = Data(
        data_generating_model,
        np_rng,
        FLAGS.num_features,
        FLAGS.num_samples,
        FLAGS.sigma_noise,
    )

    model = Model(tf_rng, FLAGS.num_features)
    logging.debug(model.model)

    optimizer = tf.optimizers.SGD(learning_rate=FLAGS.learning_rate)

    bar = trange(FLAGS.num_iters)
    for i in bar:
        with tf.GradientTape() as tape:
            x, y = data.get_batch(np_rng, FLAGS.batch_size)
            y_hat = model(x)
            loss = 0.5 * tf.reduce_mean((y_hat - y) ** 2)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        bar.set_description(f"Loss @ {i} => {loss.numpy():0.6f}")
        bar.refresh()

    logging.debug(model.model)

    # print out true values versus estimates
    print("w,    w_hat")
    compare_linear_models(data.model, model.model)

    if FLAGS.num_features > 1:
        # Only continue to plotting if x is a scalar
        exit(0)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3), dpi=200)

    ax[0].set_title("Sinewave Regression")
    ax[0].set_xlabel("x")
    ylab = ax[0].set_ylabel("y", labelpad=10)
    ylab.set_rotation(0)

    xs = np.linspace(.1,.9, 1000)
    xs = xs[: np.newaxis]
    y_hat = model(xs)
    ax[0].plot(xs,np.squeeze(y_hat), "--", color="red") 
    ax[0].plot(np.squeeze(data.x), data.y,"o", color="blue")
    

    real_y = np.sin(2*np.pi*xs)
    ax[0].plot(xs,real_y,color="green")


    ax[1].set_title("Basis Functions")
    ax[1].set_xlabel("x")
    ylab = ax[1].set_ylabel("y",labelpad=10)
    ylab.set_rotation(0)

    phi = np.transpose(np.exp((-(xs.T - model.mew) ** 2) / (model.sigma) ** 2  ))
    ax[1].plot(xs,phi)

    plt.tight_layout()
    plt.savefig(f"{script_path}/fit.pdf")


if __name__ == "__main__":
    app.run(main)
