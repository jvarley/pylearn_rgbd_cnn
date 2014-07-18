
import h5py
import os

from pylearn2.datasets import preprocessing

import hdf5_data_preprocessor

PYLEARN_DATA_PATH = os.environ["PYLEARN2_DATA_PATH"]


def preprocess_nyu_depth_dataset(which_set='train'):
    if which_set == 'train':
        num_patches = 100000
    elif which_set == 'valid':
        num_patches = 1000
    elif which_set == 'test':
        num_patches = 1000

    # this is the downloaded nyu data
    raw_hdf5_dataset_filepath = PYLEARN_DATA_PATH + "/nyu_depth_labeled/nyu_depth_v2_labeled.mat"

    #want to do multiple things to this dataset, so lets run it through
    #a pipeline
    pipeline = preprocessing.Pipeline()

    # first pull the data out of the raw_datafile and format it
    # so that we can pass it into the HDF5Dataset constructor
    # rgbd : (1449, 640, 480, 4)
    # labels: (1449, 640, 480)
    pipeline.items.append(hdf5_data_preprocessor.ExtractRawNYUData(raw_hdf5_dataset_filepath,
                                                                   data_labels=("rgbd", "labels")))

    # Next we want to pull out small patches for training
    # rgbd_patches = (num_patches, patch_shape, 4)
    # patch_labels = (num_patches, 1)
    pipeline.items.append(hdf5_data_preprocessor.ExtractPatches(patch_shape=(25, 25), num_patches=num_patches))

    # Next we want to flatten patches as it just seems to be handled better,
    # and nont of the tools for viewing topological data are working well
    # for rgbd data
    # rgbd_flattened_patches : (num_patches, patch_shape[0]*patch_shape[1]*4)
    pipeline.items.append(hdf5_data_preprocessor.FlattenPatches(patch_label="rgbd_patches",
                                                                flattened_patch_label="rgbd_flattened_patches"))
    #now we want to normalize the patches
    pipeline.items.append(
        hdf5_data_preprocessor.GlobalContrastNormalizePatches(
            data_to_normalize_key="rgbd_flattened_patches",
            batch_size=100,
            subtract_mean=True,
            scale=1.,
            sqrt_bias=0.,
            use_std=False,
            min_divisor=1e-8
        )
    )

    #now lets actually make a new dataset and run it through the pipeline
    hd5f_dataset_filepath = PYLEARN_DATA_PATH + "/nyu_depth_labeled/" + which_set + ".mat"
    hd5f_dataset = h5py.File(hd5f_dataset_filepath)
    pipeline.apply(hd5f_dataset)


if __name__ == "__main__":

    sets = ["train", "valid", "test"]
    for set_name in sets:
        print "creating " + set_name + " dataset"
        preprocess_nyu_depth_dataset(which_set=set_name)
