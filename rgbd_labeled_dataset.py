
from pylearn2.datasets import dense_design_matrix
import numpy as np

import h5py
import cPickle
import pylearn2.utils.one_hot

def get_dataset(which_set='train', one_hot=1, start=0, stop=10):
    f = open("cnn_rgbd_preprocessed_train_dataset.pkl")
    dataset = cPickle.load(f)
    return dataset


class RGBDLabeledDataset(dense_design_matrix.DenseDesignMatrix):

    def __init__(self, datafile, which_set="train"):
        """
        Helper for constructor to loadX
        In [54]: rgbd.shape
        Out[54]: (1449, 640, 480, 4)
        """
        def load_rgbd_data(data, samples_range=(0, 1449)):

            depth = data['depths'][samples_range[0]:samples_range[1], :, :]
            rgb = data['images'][samples_range[0]:samples_range[1], :, :, :]
            depth_expanded = np.expand_dims(depth, 1)

            print "concatenating rgb and depth, this could take several minutes"
            rgbd = np.concatenate((rgb, depth_expanded), axis=1)

            #at this point rgbd.shape = (1449, 4, 640, 480),
            #this will change change shape to (1449, 640, 480, 4)
            rgbd = np.rollaxis(rgbd, 1, 4)

            if rgbd.shape != (samples_range[1]-samples_range[0], 640, 480, 4):
                raise ValueError("rgbd shape is wrong!.  "
                                 "Should be (" + str(samples_range[1]-samples_range[0]) + ", 640,480,4), but is:"
                                 + str(rgbd.shape))

            return rgbd

        """
        Helper for constructor to loady
        In [54]: labels.shape
        Out[54]: (1449, 640, 480, 1)
        """
        def load_labels(data, samples_range=(0, 1449)):

            labels = data['labels'][samples_range[0]:samples_range[1], :, :]
            labels = np.expand_dims(labels, 3)
            # import IPython
            # IPython.embed()
            #labels_hot = pylearn2.utils.one_hot.one_hot(labels=labels.flatten(), max_label=894)

            #return labels_hot
            return labels

        data = h5py.File(datafile, "r")
        num_images = 50
        if which_set == "train":
            samples_range = (0, num_images)
        else:
            samples_range = (0, num_images)

        topo_x = load_rgbd_data(data, samples_range)
        topo_y = load_labels(data, samples_range)

        super(RGBDLabeledDataset, self).__init__(topo_view=topo_x, y=topo_y)



