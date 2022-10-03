import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense,\
    Flatten, Dropout, ReLU, BatchNormalization, Input, ZeroPadding2D, AveragePooling2D
import tensorflow_addons as tfa

FILE_NAME = "data_batch_"
FILE_COUNT = 5
TEST_FILE = "test_batch"
META_FILE = "batches.meta"
RNG_SEED = 31415926
LEARNING_RATE = .001
BATCH_SIZE = 128
DEFAULT_EPOCHS = 30


def get_meta(file: str):
    return unpickle(file)[b'label_names']


def load_batch(file: str, label_key: str = b'labels'):
    dict = unpickle(file)
    return dict[b'data'].reshape(len(dict[label_key]), 3, 32, 32).transpose(0, 2, 3, 1),\
        np.array(tf.keras.utils.to_categorical(dict[label_key]))


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


def i_skip(x, filter):
    x_skip = x
    x = Conv2D(filter, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = ReLU()(x)
    x = Conv2D(filter, (3, 3), padding='same')(x)
    x = Conv2D(filter, (3, 3), padding='same')(x)
    x = Dropout(.4)(x)
    x = Conv2D(filter, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = x + x_skip
    return ReLU()(x)


def conv_skip(x, filter):
    x_skip = x
    x = Conv2D(filter, (3, 3), padding='same', strides=(2, 2))(x)
    x = BatchNormalization(axis=3)(x)
    x = ReLU()(x)
    x = Conv2D(filter, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x_skip = Conv2D(filter, (1, 1), strides=(2, 2))(x_skip)
    x = x + x_skip
    return ReLU()(x)


def model_builder(input_shape: tuple, classes: int):
    filter_size = 64
    x_input = Input(input_shape)
    x = ZeroPadding2D((3, 3))(x_input)
    x = Conv2D(filter_size, kernel_size=7, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    block_layers = [3, 4, 6, 3]

    for i, layer in enumerate(block_layers):
        if i == 0:
            # skip layer
            for _ in range(layer):
                x = i_skip(x, filter_size)
        else:
            filter_size = filter_size * 2
            x = conv_skip(x, filter_size)
            for _ in range(layer - 1):
                x = i_skip(x, filter_size)

    x = AveragePooling2D((2, 2), padding='same')(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(classes, activation='softmax')(x)
    return tf.keras.models.Model(inputs=x_input, outputs=x)


def cifar100(EPOCHS: int = DEFAULT_EPOCHS):
    seed_sequence = np.random.SeedSequence(RNG_SEED)
    [np_seed] = seed_sequence.spawn(1)
    np_rng = np.random.default_rng(np_seed)
    x, y = load_batch("cifar-100-python/train", b'fine_labels')
    indexes = np_rng.permutation(len(y))
    x_r = x[indexes]
    y_r = y[indexes]
    split = int(.8*len(y))
    x_train = x_r[0:split]
    y_train = y_r[0:split]
    x_val = x_r[split::]
    y_val = y_r[split::]

    model = model_builder((32, 32, 3), 128)

    optimizer = tfa.optimizers.AdamW(
        weight_decay=1e-6, learning_rate=LEARNING_RATE)
    model.summary()
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy', metrics=['accuracy'])
    _ = model.fit(x_train, y_train, epochs=EPOCHS,
                  batch_size=BATCH_SIZE, validation_data=(x_val, y_val))

    x_test, y_test = load_batch("cifar-100-python/test", b'fine_labels')
    _, acc = model.evaluate(x_test, y_test)
    print("acc: " + str(acc))


def cifar10(EPOCHS: int = DEFAULT_EPOCHS):
    FOLDER_NAME = "data/"
    # names = get_meta(FOLDER_NAME + META_FILE)
    x_train, y_train, x_val, y_val = load_data(
        FOLDER_NAME + FILE_NAME, FILE_COUNT)

    model = model_builder((32, 32, 3), 10)
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


def main():
    cifar10(20)


if __name__ == "__main__":
    main()
