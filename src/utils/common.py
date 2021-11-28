import yaml
import time

def read_config_file(config_path):
    with open(config_path) as config_file:
        content = yaml.safe_load(config_file)

    return content


def get_unique_filename(filename):
    unique_filename = time.strftime(f"%Y%m%d_%H%M%S_{filename}")
    return unique_filename


def get_timestamp_filename(filename):
    timestamp = time.asctime().replace(' ', '_').replace(':', '')
    unique_filename = f"{filename}_at_{timestamp}"
    return unique_filename