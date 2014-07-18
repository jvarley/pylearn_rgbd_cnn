
import os
from pylearn2.testing import skip
from pylearn2.config import yaml_parse


def train_convolutional_network():

    skip.skip_if_no_data()
    yaml_file_path = os.path.abspath(os.path.dirname(__file__))
    save_path = os.path.dirname(os.path.realpath(__file__))

    yaml = open("{0}/conv_training_model.yaml".format(yaml_file_path), 'r').read()
    hyper_params = {'batch_size': 50,
                    'output_channels_h2': 32,
                    'output_channels_h3': 64,
                    'max_epochs': 500,
                    'save_path': save_path}

    yaml = yaml % hyper_params
    train = yaml_parse.load(yaml)

    train.main_loop()

if __name__ == "__main__":
    train_convolutional_network()
