from datasets import load_dataset
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, ReLU, Input, Embedding, Bidirectional, LSTM, GlobalMaxPool1D

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
    x_train = convert_to_sequence(x_raw, tokenizer)
    y_train = tf.keras.utils.to_categorical(y)
    x_test_f, y_test_f = build_data(test)
    x_test_ft = convert_to_sequence(x_test_f, tokenizer)
    y_test_ft = tf.keras.utils.to_categorical(y)
    x_test, y_test, x_validate, y_validate = split(x_test_ft, y_test_ft, np_rng)

    model = Sequential([
        Embedding(input_dim=MAX_SIZE, output_dim=2048),
        Bidirectional(LSTM(128, return_sequences=True)),
        Bidirectional(LSTM(128, return_sequences=True)),
        Bidirectional(LSTM(128, return_sequences=True)),
        GlobalMaxPool1D(),
        Dense(512, activation='relu'),
        Dense(512, activation='relu'),
        Dropout(.5),
        Dense(num_classes, activation="softmax")
    ])
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    _ = model.fit(x_train, y_train, epochs=10, batch_size=BATCH_SIZE, validation_data=(x_validate, y_validate))


    _, acc = model.evaluate(x_test, y_test)
    print("acc: " + str(acc))


if __name__ == "__main__":
    main()

"""
I had a lot of issues with this model, This model gets an accuracy of 25% due to overfitting. I tried 
using Bert as an embedding layer, but I had issues getting it working and ran out of time. 
"""
