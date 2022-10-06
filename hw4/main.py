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


def res50():
    input = Input(shape=(32, 32, 3))
    x = ZeroPadding2D(padding=(3, 3))(input)
    x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    #2nd stage 
    # frm here on only conv block and identity block, no pooling

    x = skip_conv(x, s=1, filters=(64, 256))
    x = skip_identity(x, filters=(64, 256))
    x = skip_identity(x, filters=(64, 256))

    # 3rd stage

    x = skip_conv(x, s=2, filters=(128, 512))
    x = skip_identity(x, filters=(128, 512))
    x = skip_identity(x, filters=(128, 512))
    x = skip_identity(x, filters=(128, 512))

    # 4th stage

    x = skip_conv(x, s=2, filters=(256, 1024))
    x = skip_identity(x, filters=(256, 1024))
    x = skip_identity(x, filters=(256, 1024))
    x = skip_identity(x, filters=(256, 1024))
    x = skip_identity(x, filters=(256, 1024))
    x = skip_identity(x, filters=(256, 1024))

    # 5th stage

    x = skip_conv(x, s=2, filters=(512, 2048))
    x = skip_identity(x, filters=(512, 2048))
    x = skip_identity(x, filters=(512, 2048))

    # ends with average pooling and dense connection

    x = AveragePooling2D((2, 2), padding='same')(x)

    x = Flatten()(x)
    x = Dense(100, activation='softmax', kernel_initializer='he_normal')(x)

    return tf.keras.models.Model(inputs = input, outputs=x)



def skip_conv(x, s, filters):
    x_skip = x
    f1, f2 = filters
    l2 = tf.keras.regularizers.L2

    # first block
    x = Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x)
    # when s = 2 then it is like downsizing the feature map
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # second block
    x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # third block
    x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)

    # shortcut
    x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x_skip)
    x_skip = BatchNormalization()(x_skip)

    # add
    x = x + x_skip
    x = ReLU()(x)

    return x

def skip_identity(x, filters):
    x_skip = x
    f1, f2 = filters
    l2 = tf.keras.regularizers.L2

    # First Block
    x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Second Block
    x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # third block activation used after adding the input
    x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)

    # add the input
    x = x + x_skip
    x = ReLU()(x)

    return x


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

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(.2),
        Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(.3),
        Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(.4),
        Flatten(),
        Dense(100, activation="softmax")
    ])

    model.summary()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.metrics.TopKCategoricalAccuracy(5)])
    _ = model.fit(x_train, y_train, epochs=EPOCHS,
                  batch_size=BATCH_SIZE, validation_data=(x_val, y_val))

    x_test, y_test = load_batch("cifar-100-python/test", b'fine_labels')
    _, acc, top5 = model.evaluate(x_test, y_test)
    print("acc: " + str(acc))
    print("top5: " + str(top5))


def cifar10(EPOCHS: int = DEFAULT_EPOCHS):
    FOLDER_NAME = "data/"
    # names = get_meta(FOLDER_NAME + META_FILE)
    x_train, y_train, x_val, y_val = load_data(
        FOLDER_NAME + FILE_NAME, FILE_COUNT)

    # model = model_builder((32, 32, 3), 10)
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(.2),
        Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(.3),
        Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(.4),
        Flatten(),
        Dense(10, activation="softmax")
        ])
    # optimizer = tfa.optimizers.AdamW(
    #     weight_decay=1e-6, learning_rate=LEARNING_RATE)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=.1)
    model.summary()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=EPOCHS,
                        batch_size=BATCH_SIZE, validation_data=(x_val, y_val))

    x_test, y_test = load_batch(FOLDER_NAME + TEST_FILE)
    _, acc = model.evaluate(x_test, y_test)
    print("acc: " + str(acc))


def main():
    cifar100(50)


if __name__ == "__main__":
    main()
