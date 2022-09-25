import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense,\
    Flatten, Dropout, ReLU, BatchNormalization
import tensorflow_addons as tfa

FOLDER_NAME = "data/"
FILE_NAME = "data_batch_"
FILE_COUNT = 5
TEST_FILE = "test_batch"
META_FILE = "batches.meta"
RNG_SEED = 31415926
LEARNING_RATE = .001
BATCH_SIZE = 128
EPOCHS = 30


def get_meta(file: str):
    return unpickle(file)[b'label_names']


def load_batch(file: str):
    dict = unpickle(file)
    return dict[b'data'].reshape(len(dict[b'labels']), 3, 32, 32).transpose(0, 2, 3, 1),\
        np.array(tf.keras.utils.to_categorical(dict[b'labels']))


def unpickle(file: str):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_data(file_base: str, batches: int):
    x = []
    y = []
    # well I can't vectorize IO operations so here is a gross for loop
    for i in range(batches):
        x_t, y_t = load_batch(file_base + str(i+1))
        x.append(x_t)
        y.append(y_t)
    return np.concatenate(x[0:-1]), np.concatenate(y[0:-1]), x[-1], y[-1]


def plotImg(img: np.ndarray, label: str):
    plt.imshow(img)
    plt.title(label)
    plt.show()


def main():
    # names = get_meta(FOLDER_NAME + META_FILE)
    x_train, y_train, x_val, y_val = load_data(
        FOLDER_NAME + FILE_NAME, FILE_COUNT)

    model = Sequential([
        Conv2D(256, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        ReLU(),
        Conv2D(512, (3, 3),  activation='relu'),
        Conv2D(512, (3, 3),  activation='relu'),
        Conv2D(512, (3, 3),  activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(512, (3, 3), activation='relu'),
        Conv2D(512, (3, 3), activation='relu'),
        Conv2D(512, (3, 3), activation='relu'),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(256, activation='relu'),
        Dropout(.3),
        Dense(256, activation='relu'),
        Dense(256, activation='relu'),
        Dropout(.2),
        Dense(10, activation='softmax', kernel_regularizer='l2')
    ])

    optimizer = tfa.optimizers.AdamW(
        weight_decay=1e-6, learning_rate=LEARNING_RATE)
    model.summary()
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=EPOCHS,
                        batch_size=BATCH_SIZE, validation_data=(x_val, y_val))

    x_test, y_test = load_batch(FOLDER_NAME + TEST_FILE)
    _, acc = model.evaluate(x_test, y_test)
    print("acc: " + str(acc))


if __name__ == "__main__":
    main()
