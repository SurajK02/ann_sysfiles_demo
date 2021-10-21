import yaml


def read_config_file(config_path):
    with open(config_path) as config_file:
        content = yaml.safe_load(config_file)

    return content
