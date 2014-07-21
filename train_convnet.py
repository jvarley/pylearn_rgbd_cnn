
import os
from pylearn2.testing import skip
from pylearn2.config import yaml_parse


def get_25_25_model(yaml_path, save_path):

    yaml = open("{0}/conv_model.yaml".format(yaml_path), 'r').read()
    hyper_params = {'batch_size': 50,
                    'output_channels_h2': 32,
                    'output_channels_h3': 64,
                    'max_epochs': 500,
                    'save_path': save_path}

    yaml_with_hyper_params = yaml % hyper_params
    return yaml_with_hyper_params


def get_72_72_model(yaml_path, save_path):

    yaml = open("{0}/conv_model2.yaml".format(yaml_path), 'r').read()
    hyper_params = {'batch_size': 50,
                    'output_channels_h2': 32,
                    'output_channels_h3': 64,
                    'output_channels_h4': 128,
                    'max_epochs': 500,
                    'save_path': save_path}

    yaml_with_hyper_params = yaml % hyper_params
    return yaml_with_hyper_params


def train_convolutional_network(yaml_with_hyper_params):

    skip.skip_if_no_data()

    train = yaml_parse.load(yaml_with_hyper_params)

    train.main_loop()


if __name__ == "__main__":
    yaml_file_path = os.path.abspath(os.path.dirname(__file__)) + "/models"
    model_save_path = os.path.dirname(os.path.realpath(__file__)) + "/models"

    model_yaml = get_72_72_model(yaml_file_path, model_save_path)
    #model_yaml = get_25_25_model(yaml_file_path, model_save_path)

    train_convolutional_network(model_yaml)
