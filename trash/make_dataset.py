
import os

# We'll need the serial module to save the dataset
from pylearn2.utils import serial

# We'll need the preprocessing module to preprocess the dataset
from pylearn2.datasets import preprocessing

#my patch extractor that also extracts a y for each patch
import rgbd_preprocessor
import argparse

import numpy as np
import h5py
import rgbd_labeled_dataset
import rgbd_labeled_dataset_hdf5
import copy

pylearn_data_path = os.environ["PYLEARN2_DATA_PATH"]
num_images = 1449

def build_raw_dataset():

    datafile = pylearn_data_path + "/nyu_depth_labeled/nyu_depth_v2_labeled.mat"
    infile = h5py.File(datafile)

    outfile = h5py.File(pylearn_data_path + "/nyu_depth_labeled/out.mat" , 'a')

    #outfile["rgbd"] = np.zeros((num_images, 640, 480, 4))
    outfile.create_dataset("rgbd", (num_images, 640, 480, 4), chunks=(10, 640, 480, 4))
    outfile.create_dataset("labels", (num_images, 640, 480), chunks=(10, 640, 480))

    for i in range(num_images):
        if i%100 == 0:
            print str(i) + "/" + str(num_images)

        outfile["rgbd"][i, :, :, 0:3] = np.rollaxis(infile["images"][i], 0, 3)
        outfile["rgbd"][i, :, :, 3] = infile["depths"][i]
        outfile["labels"][i] = infile["labels"][i]


def build_dataset_patches():
    num_patches = 100000
    patch_shape = (60,80)
    #just rows and columns
    num_topological_dimensions = 2
    h5py_file = h5py.File(pylearn_data_path + "/nyu_depth_labeled/out.mat", 'a')
    h5py_file.create_dataset("rgbd_patches", (num_patches, patch_shape[0], patch_shape[1], 4), chunks=(100, patch_shape[0], patch_shape[1], 4))
    h5py_file.create_dataset("patch_labels", (num_patches, 1))

    import IPython
    IPython.embed()
    X = h5py_file["rgbd"]
    y = h5py_file["labels"]

    patch_X = h5py_file["rgbd_patches"]
    patch_y = h5py_file["patch_labels"]

    channel_slice = slice(0, X.shape[-1])

    rng = preprocessing.make_np_rng([1, 2, 3], which_method="randint")

    for i in xrange(num_patches):

        image_num = rng.randint(num_images)
        x_args = [image_num]
        y_args = [image_num]

        for j in xrange(num_topological_dimensions):
            max_coord = X.shape[j + 1] - patch_shape[j]
            coord = rng.randint(max_coord + 1)
            x_args.append(slice(coord, coord + patch_shape[j]))
            y_args.append(coord + patch_shape[j]/2.0)

        x_args.append(channel_slice)

        patch_X[i] = X[tuple(x_args)]
        patch_y[i] = y[tuple(y_args)]



def preprocess_h5py_dataset():

    pylearn_data_path = os.environ["PYLEARN2_DATA_PATH"]
    datafile = pylearn_data_path + "/nyu_depth_labeled/out.mat"

    dataset = rgbd_labeled_dataset_hdf5.HDF5Dataset(datafile, X="rgbd", y="labels")

    # We'd like to do several operations on them, so we'll set up a pipeline to
    # do so.
    pipeline = preprocessing.Pipeline()

    # First we want to pull out small patches of the images, since it's easier
    # to train an RBM on these
    pipeline.items.append(
        rgbd_preprocessor.ExtractPatches(patch_shape=(60, 80), num_patches=100000)
    )

    # Next we contrast normalize the patches. The default arguments use the
    # same "regularization" parameters as those used in Adam Coates, Honglak
    # Lee, and Andrew Ng's paper "An Analysis of Single-Layer Networks in
    # Unsupervised Feature Learning"
    pipeline.items.append(preprocessing.GlobalContrastNormalization(sqrt_bias=10., use_std=True, batch_size=500))

    # Finally we whiten the data using ZCA. Again, the default parameters to
    # ZCA are set to the same values as those used in the previously mentioned
    # paper.
    zca = preprocessing.ZCA()
    zca.set_matrices_save_path("data/zca_matrix_save_path")
    pipeline.items.append(zca)

    # Here we apply the preprocessing pipeline to the dataset. The can_fit
    # argument indicates that data-driven preprocessing steps (such as the ZCA
    # step in this example) are allowed to fit themselves to this dataset.
    # Later we might want to run the same pipeline on the test set with the
    # can_fit flag set to False, in order to make sure that the same whitening
    # matrix was used on both datasets.
    dataset.apply_preprocessor(preprocessor=pipeline, can_fit=True)

    # Finally we save the dataset to the filesystem. We instruct the dataset to
    # store its design matrix as a numpy file because this uses less memory
    # when re-loading (Pickle files, in general, use double their actual size
    # in the process of being re-loaded into a running process).
    # The dataset object itself is stored as a pickle file.
    #path = pylearn2.__path__[0]
    train_example_path = os.getcwd()
    #dataset.use_design_loc(os.path.join(train_example_path, "h5py_design_dataset"))

    #train_pkl_path = os.path.join(train_example_path, dataset_file)
    #serial.save(train_pkl_path, dataset)

def preprocess_pickle_dataset():

    parser = argparse.ArgumentParser()
    parser.add_argument('--design_dataset', nargs='?', const="bar", default='data/cnn_rgbd_preprocessed_train_design_dataset.npy')
    parser.add_argument('--dataset', nargs='?', const="bar", default='data/cnn_rgbd_preprocessed_train_dataset.pkl')
    args = parser.parse_args()

    design_dataset_file = args.design_dataset
    dataset_file = args.dataset


    pylearn_data_path = os.environ["PYLEARN2_DATA_PATH"]
    datafile = pylearn_data_path + "/nyu_depth_labeled/nyu_depth_v2_labeled.mat"

    dataset = rgbd_labeled_dataset.RGBDLabeledDataset(datafile, "train")

    # We'd like to do several operations on them, so we'll set up a pipeline to
    # do so.
    pipeline = preprocessing.Pipeline()

    # First we want to pull out small patches of the images, since it's easier
    # to train an RBM on these
    pipeline.items.append(
        rgbd_preprocessor.ExtractPatches(patch_shape=(60, 80), num_patches=100000)
    )

    # Next we contrast normalize the patches. The default arguments use the
    # same "regularization" parameters as those used in Adam Coates, Honglak
    # Lee, and Andrew Ng's paper "An Analysis of Single-Layer Networks in
    # Unsupervised Feature Learning"
    pipeline.items.append(preprocessing.GlobalContrastNormalization(sqrt_bias=10., use_std=True, batch_size=500))

    # Finally we whiten the data using ZCA. Again, the default parameters to
    # ZCA are set to the same values as those used in the previously mentioned
    # paper.
    zca = preprocessing.ZCA()
    zca.set_matrices_save_path("data/zca_matrix_save_path")
    pipeline.items.append(zca)

    # Here we apply the preprocessing pipeline to the dataset. The can_fit
    # argument indicates that data-driven preprocessing steps (such as the ZCA
    # step in this example) are allowed to fit themselves to this dataset.
    # Later we might want to run the same pipeline on the test set with the
    # can_fit flag set to False, in order to make sure that the same whitening
    # matrix was used on both datasets.
    dataset.apply_preprocessor(preprocessor=pipeline, can_fit=True)

    # Finally we save the dataset to the filesystem. We instruct the dataset to
    # store its design matrix as a numpy file because this uses less memory
    # when re-loading (Pickle files, in general, use double their actual size
    # in the process of being re-loaded into a running process).
    # The dataset object itself is stored as a pickle file.
    #path = pylearn2.__path__[0]
    train_example_path = os.getcwd()
    dataset.use_design_loc(os.path.join(train_example_path, design_dataset_file))

    train_pkl_path = os.path.join(train_example_path, dataset_file)
    serial.save(train_pkl_path, dataset)


if __name__ == "__main__":
    build_raw_dataset()
    build_dataset_patches()