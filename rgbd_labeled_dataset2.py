from pylearn2.datasets.hdf5 import HDF5Dataset
import h5py
import numpy as np

import IPython
class RGBDLabeledDataset(HDF5Dataset):

    def __init__(self, filename, X=None, topo_view=None, y=None,
             load_all=True, which_set="train", one_hot=None,
             start=None,  stop=None, **kwargs):
        self.load_all = load_all

        """
        In [54]: rgbd.shape
        Out[54]: (1449, 4, 640, 480)
        """
        def load_rgbd_data(samples_range=(0, 1449)):
            depth = self._file['depths'][samples_range[0]:samples_range[1], :, :]
            rgb = self._file['images'][samples_range[0]:samples_range[1], :, :, :]
            depth_expanded = np.expand_dims(depth, 1)

            print "concatenating rgb and depth, this could take several minutes"
            rgbd =  np.concatenate((rgb, depth_expanded), axis=1)

            return rgbd

        """
        In [54]: labels.shape
        Out[54]: (1449, 640, 480)
        """
        def load_labels(samples_range=(0, 1449)):

            labels = self._file['labels'][samples_range[0]:samples_range[1], :, :]
            return labels

        if which_set == "train":
            samples_range = (0, 100)
        else:
            samples_range = (0, 100)

        self._file = h5py.File(filename)

        X = load_rgbd_data(samples_range)
        y = load_labels(samples_range)

        if topo_view is not None:
            topo_view = self.get_dataset(topo_view, load_all)

        super(HDF5Dataset, self).__init__(X=X, topo_view=topo_view, y=y,
                                          **kwargs)