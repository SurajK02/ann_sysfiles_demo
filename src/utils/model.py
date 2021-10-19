import tensorflow as tf
import logging
import os
import time

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, "running_logs"), level=logging.INFO, format=logging_str, filemode='a')

def create_ann_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES):
    model_clf = tf.keras.models.Sequential()
    model_clf.add(tf.keras.layers.Flatten(input_shape=[28,28], name="inputLayer"))
    model_clf.add(tf.keras.layers.Dense(300, activation="relu",  name="hiddenLayer1"))
    model_clf.add(tf.keras.layers.Dense(100, activation="relu",  name="hiddenLayer2"))
    model_clf.add(tf.keras.layers.Dense(NUM_CLASSES, activation="softmax",  name="outputLayer"))

    print(model_clf.summary())
    model_clf.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=METRICS)

    return model_clf



def save_model(model, model_name, model_dir):
    filename = time.strftime("%Y%m%dT%H:%M:%S")+f"_{model_name}"
    path = os.path.join(model_dir, filename)
    model.save(path)