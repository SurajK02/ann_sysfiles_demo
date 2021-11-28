import argparse
import logging
import os

from utils.common import read_config_file, get_unique_filename
from utils.data_mgmt import load_mnist_data, normalize_images
from utils.model import create_ann_model, save_model, plot_history, get_logs_path
from utils.callbacks import get_callback_list

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, "running_logs"), level=logging.INFO, format=logging_str,
                    filemode='a')


def training(config_path, tb_logs):
    config = read_config_file(config_path)
    validation_size = config['params']['validation_size']

    logging.info(">>>>> load data >>>>>")
    X_valid, X_train, y_valid, y_train, X_test, y_test = load_mnist_data(validation_size)
    logging.info(">>>>> load data success >>>>>")

    logging.info(">>>>> normalize images >>>>>")
    X_valid = normalize_images(X_valid)
    X_train = normalize_images(X_train)
    X_test = normalize_images(X_test)
    logging.info(">>>>> image normalization success >>>>>")

    logging.info(">>>>> init model >>>>>")
    LOSS_FUNCTION = config['params']['loss_function']
    OPTIMIZER = config['params']['optimizer']
    METRICS = config['params']['metrics']
    NUM_CLASSES = config['params']['no_classes']
    model = create_ann_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES)
    logging.info(">>>>> model init success >>>>>")

    
    EPOCHS = config['params']['epochs']
    VALIDATION = (X_valid, y_valid)
    CALLBACK_LIST = get_callback_list(config, X_train, tb_img_summary=True)

    logging.info(">>>>> train model >>>>>")
    history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=VALIDATION, callbacks=CALLBACK_LIST)
    logging.info(">>>>> model training success >>>>>")

    model_name = config['artifacts']['model_name']
    model_dir = config['artifacts']['model_dir']
    artifacts_dir = config['artifacts']['artifacts_dir']
    plot_dir = config['artifacts']['plot_dir']
    log_dir = config['logs']['logs_dir']

    unique_filename = get_unique_filename(model_name)

    model_path = os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_path, exist_ok=True)
    save_model(model=model, unique_filename=unique_filename, model_dir=model_path)

    plot_path = os.path.join(artifacts_dir, plot_dir)
    os.makedirs(plot_path, exist_ok=True)
    plot_history(history, plot_path, unique_filename)

    

if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config.yaml")
    args.add_argument("--tb_logs", "-tbl", default=True)

    parsed_args = args.parse_args()

    training(config_path=parsed_args.config, tb_logs=parsed_args.tb_logs)
