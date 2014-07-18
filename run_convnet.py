import cPickle
import numpy as np
import theano
import theano.tensor as T
import os
from rgbd_labeled_dataset_hdf5 import HDF5Dataset

import h5py

pylearn_data_path = os.environ["PYLEARN2_DATA_PATH"]
orig_file_filename = pylearn_data_path + "/nyu_depth_labeled/" + "nyu_depth_v2_labeled" + ".mat"
orig_file = h5py.File(orig_file_filename)


def get_pixel_classifier():
    f = open('convolutional_network_best.pkl')
    cnn_model = cPickle.load(f)

    X = cnn_model.get_input_space().make_theano_batch()
    Y = cnn_model.fprop(X)
    Y = T.argmax(Y, axis=1)
    f = theano.function([X], Y)
    return f


def get_image():
    pylearn_data_path = os.environ["PYLEARN2_DATA_PATH"]
    hdf5_dataset_filename = pylearn_data_path + "/nyu_depth_labeled/" + "train" + ".mat"
    dataset = h5py.File(hdf5_dataset_filename)
    image = dataset['rgbd'][0]
    return image


def get_pixel_label(x, y):
    pixel_label = ""
    for i in orig_file[orig_file['names'][0][output[x,y]]][:]:
        pixel_label += chr(i)
    print pixel_label


classifier = get_pixel_classifier()
image = get_image()
image = np.expand_dims(image,0)
output = np.zeros((640, 480))

for x in range(len(image[0])):
    for y in range(len(image[0,0])):
        if x > 300 and y > 30 and x < 330 and y < 450:

            patch = image[:,x-40:x+40, y-30:y+30, :]
            output[x, y] = classifier(patch)[0]
        else:
            output[x, y] = 0

    print "x: " + str(x) + " y: " + str(y) + " out: " + str(output[x,y])


import IPython
IPython.embed()


