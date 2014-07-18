import cPickle
import numpy as np
import theano
import theano.tensor as T
import os
import h5py

import pylearn2
import matplotlib.pyplot as plt

PYLEARN_DATA_PATH = os.environ["PYLEARN2_DATA_PATH"]

CONV_MODEL_FILENAME = 'convolutional_2_layer_network_best.pkl'


def get_classifier():
    f = open(CONV_MODEL_FILENAME)
    cnn_model = cPickle.load(f)
    cnn_model = cnn_model.layers[2]

    X = cnn_model.get_input_space().make_theano_batch()
    W = cnn_model.W
    b = cnn_model.b
    Z = T.dot(X, W) + b
    Y = T.argmax(Z, axis=3)
    f = theano.function([X], Y)
    return f


def get_image_features():
    f = open(CONV_MODEL_FILENAME)
    cnn_model = cPickle.load(f)
    new_space = pylearn2.space.Conv2DSpace((320, 240), num_channels=4, axes=('b', 0, 1, 'c'), dtype='float32')

    cnn_model.layers = cnn_model.layers[0:2]
    cnn_model.layers[0].border_mode = "full"
    cnn_model.layers[1].border_mode = "full"

    cnn_model.set_input_space(new_space)
    X = cnn_model.get_input_space().make_theano_batch()
    Y = cnn_model.fprop(X)

    f = theano.function([X], Y)
    return f


def get_image(image_id=0):
    hdf5_dataset_filename = PYLEARN_DATA_PATH + "/nyu_depth_labeled/" + "train" + ".mat"
    dataset = h5py.File(hdf5_dataset_filename)
    rgbd_image = dataset['rgbd'][image_id]
    return np.expand_dims(rgbd_image, 0)


if __name__ == "__main__":
    classifier = get_classifier()
    feature_extractor = get_image_features()
    image = get_image()

    image_features = feature_extractor(image[:, 0:320, 0:240, :])
    classified_image = classifier(image_features)

    plt.imshow(classified_image[0, :, :])

    import IPython
    IPython.embed()





