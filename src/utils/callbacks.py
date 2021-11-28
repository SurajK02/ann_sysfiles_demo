import tensorflow as tf
import numpy as np
import time
import os
from utils.model import get_logs_path
from utils.common import get_timestamp_filename


def get_callback_list(config, X_train, tb_img_summary=True):

    log_dir = config['logs']['logs_dir']
    tf_logs_dir = config['logs']['tensorboard_logs']
    unique_dir_name = get_timestamp_filename("tb_logs")

    TENSORBOARD_ROOT_LOG_DIR = get_logs_path(log_folder=log_dir,log_dir=tf_logs_dir,unique_filename=unique_dir_name)
    os.makedirs(TENSORBOARD_ROOT_LOG_DIR, exist_ok=True)

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_ROOT_LOG_DIR)
    
    if tb_img_summary == True:
        tb_image_summary(config, X_train, log_dir=TENSORBOARD_ROOT_LOG_DIR)
    
    patience = config['params']['patience']
    restore_best_weights = config['params']['restore_best_weights']
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=restore_best_weights)

    artifacts_dir = config['artifacts']['artifacts_dir']
    checkpoint_dir = config['artifacts']['checkpoint_dir']
    CHKPNT_DIR = os.path.join(artifacts_dir, checkpoint_dir)
    os.makedirs(CHKPNT_DIR, exist_ok=True)
    CHKPNT_PATH = os.path.join(CHKPNT_DIR, "model_chkpnt.h5")
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(CHKPNT_PATH, save_best_only=True)

    return [tensorboard_cb, early_stopping_cb, checkpoint_cb]

    
    


def tb_image_summary(config, X_train, log_dir):  

    file_writer = tf.summary.create_file_writer(logdir=log_dir)
    with file_writer.as_default():
        images = np.reshape(X_train, (-1, 28, 28, 1))
        tf.summary.image("samples of handwritten images", images, max_outputs=25, step=0)