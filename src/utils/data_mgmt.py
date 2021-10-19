import tensorflow as tf

def load_mnist_data(validation_size):
    mnist = tf.keras.datasets.mnist
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

    X_valid = X_train_full[:validation_size]
    X_train = X_train_full[validation_size:]

    y_valid = y_train_full[:validation_size]
    y_train = y_train_full[validation_size:]

    X_test = X_test

    return X_valid, X_train, y_valid, y_train, X_test, y_test


def normalize_images(data):
    return data/255