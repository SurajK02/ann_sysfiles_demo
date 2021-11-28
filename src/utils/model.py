import tensorflow as tf
import logging
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, "running_logs"), level=logging.INFO, format=logging_str,
                    filemode='a')


def create_ann_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES):
    model_clf = tf.keras.models.Sequential()
    model_clf.add(tf.keras.layers.Flatten(input_shape=[28, 28], name="inputLayer"))
    model_clf.add(tf.keras.layers.Dense(300, activation="relu", name="hiddenLayer1"))
    model_clf.add(tf.keras.layers.Dense(100, activation="relu", name="hiddenLayer2"))
    model_clf.add(tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="outputLayer"))

    print(model_clf.summary())
    model_clf.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=METRICS)

    return model_clf


def save_model(model, unique_filename, model_dir):
    unique_filename = unique_filename+".h5"
    path = os.path.join(model_dir, unique_filename)
    model.save(path)

def save_plot(plot_dir, unique_filename):
    unique_filename = unique_filename+".png"
    path = os.path.join(plot_dir, unique_filename)
    plt.savefig(path)

def plot_history(history, plot_dir, unique_filename):
    pd.DataFrame(history.history).plot(figsize=(10, 10))
    plt.grid(True)
    save_plot(plot_dir, unique_filename)

def get_logs_path(log_folder,log_dir,unique_filename):
    log_path = os.path.join(log_folder, log_dir, unique_filename)
    print(f"logs are saved at: {log_path}")
    return log_path



