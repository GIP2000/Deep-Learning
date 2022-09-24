import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten,Dropout

FOLDER_NAME = "data/"
FILE_NAME = "data_batch_"
FILE_COUNT = 5
META_FILE = "batches.meta"
RNG_SEED = 31415926
LEARNING_RATE = .001
BATCH_SIZE = 128
EPOCHS = 50


def get_meta(file):
    return unpickle(file)[b'label_names']


def load_batch(file):
    dict = unpickle(file)
    return dict[b'data'].reshape(len(dict[b'labels']),3,32,32).transpose(0,2,3,1), dict[b'labels']


def load_data(file_base, batches):
    x = []
    y = []
    for i in range(batches - 1):
        x_t, y_t = load_batch(file_base + str(i+1))
        x.append(x_t)
        y.append(tf.keras.utils.to_categorical(y_t))
    x_val, y_val = load_batch(file_base + str(batches - 1))
    return np.concatenate(x), np.concatenate(y), x_val, np.array(tf.keras.utils.to_categorical(y_val))


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def split(data: np.ndarray, label: np.array, rng):
    indexes = rng.permutation(len(label))
    r_data = data[indexes]
    r_label = label[indexes]
    validiation_split = int(.8 * len(r_label))

    return (r_data[0:validiation_split]), tf.keras.utils.to_categorical(r_data[0:validiation_split]), \
        (r_data[validiation_split::]), tf.keras.utils.to_categorical(r_label[validiation_split::])


def plotImg(img: np.ndarray, label: str):
    plt.imshow(img)
    plt.title(label)
    plt.show()


def main():
    seed_sequence = np.random.SeedSequence(RNG_SEED)
    [np_seed] = seed_sequence.spawn(1)
    np_rng = np.random.default_rng(np_seed)

    # names = get_meta(FOLDER_NAME + META_FILE)
    x_train, y_train, x_val, y_val = load_data(FOLDER_NAME + FILE_NAME, FILE_COUNT)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Dropout(.2),
        Conv2D(64, (3, 3),  activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax', kernel_regularizer='l2')
    ])

    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_val, y_val))


if __name__ == "__main__":
    main()
