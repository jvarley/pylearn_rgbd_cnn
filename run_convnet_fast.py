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

    #This is the output of the feature extractor
    #shape should be:
    #(~num_rows, ~num_cols, num_features)
    #it is ~num_rows and ~num_cols because depending on
    #pool_stride and kernel_size from the feature extraction, there is some 0-padding of the input
    # and the max pooling with scale the output down
    X = cnn_model.get_input_space().make_theano_batch()

    #Z is now a 3d tensor of shape
    # (~num_rows,~num_cols, num_labels)
    Z = T.dot(X, cnn_model.W) + cnn_model.b

    #we can take the arg max to get the per pixel label
    #So Y is of shape:
    # (1, ~num_rows, ~num_cols)
    Y = T.argmax(Z, axis=3)

    return theano.function([X], Y)


def get_feature_extractor():
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


def get_test_image(image_id=0):
    hdf5_dataset_filename = PYLEARN_DATA_PATH + "/nyu_depth_labeled/" + "train" + ".mat"
    dataset = h5py.File(hdf5_dataset_filename)
    rgbd_image = dataset['rgbd'][image_id]
    return np.expand_dims(rgbd_image, 0)


if __name__ == "__main__":
    #this is the bottom half of the trained conv net
    #we just change the shape of the input and remove the
    # fully connected top layer
    feature_extractor = get_feature_extractor()

    #this is the top half of the trained convnet
    #we need to remove the softmax functionality since
    # 1) theano does not support softmax on a 3d tensor
    # 2) we only need it for training, we can run argmax on the
    #    raw outputs of the classifier without running softmax first
    classifier = get_classifier()

    #just grab a random image to test with
    image = get_test_image()

    image_features = feature_extractor(image[:, 0:320, 0:240, :])
    classified_image = classifier(image_features)

    plt.imshow(classified_image[0, :, :])

    import IPython
    IPython.embed()





