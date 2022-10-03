from datasets import load_dataset
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, ReLU, Input

MAX_WORDS = 100000
MAX_SIZE = 1012
EPOCHS = 30
BATCH_SIZE = 128
RNG_SEED = 31415926


def build_data(data):
    value, label = data.data
    return np.array(value), np.array(label)


def split(x, y, rng):
    indexes = rng.permutation(len(y))
    r_x = x[indexes]
    r_y = y[indexes]
    sp = int(.7 * len(y))
    return r_x[0:sp], r_y[0:sp], r_x[sp::], r_y[sp::]


def convert_to_sequence(x_raw, tokenizer, len=MAX_SIZE):
    x_s = tokenizer.texts_to_sequences(x_raw)
    x = np.array(pad_sequences(x_s, maxlen=len))
    return x


def main():
    np_rng = np.random.default_rng(RNG_SEED)
    dataset = load_dataset("ag_news")

    test, train = dataset.values()
    x_raw, y = build_data(train)
    num_classes = max(y) + 1

    tokenizer = Tokenizer(oov_token='<OOV>')
    tokenizer.fit_on_texts(x_raw)
    x = convert_to_sequence(x_raw, tokenizer)
    y = tf.keras.utils.to_categorical(y)
    x_train, y_train, x_validate, y_validate = split(x, y, np_rng)
    model = Sequential([
        Input((MAX_SIZE,)),
        Dense(512, activation="relu"),
        Dense(512, activation="relu"),
        Dense(512, activation="relu"),
        Dense(512, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])
    print(x_train.shape)
    print(x_validate.shape)
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    _ = model.fit(x_train, y_train, epochs=10, batch_size=BATCH_SIZE, validation_data=(x_validate, y_validate))

    x_test_raw, y_test_raw = build_data(test)
    x_test = convert_to_sequence(x_test_raw, tokenizer)
    y_test = tf.keras.utils.to_categorical(y_test_raw)

    _, acc = model.evaluate(x_test, y_test)
    print("acc: " + str(acc))


if __name__ == "__main__":
    main()
