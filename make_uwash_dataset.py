
import h5py
import os

from pylearn2.datasets import preprocessing

import hdf5_data_preprocessors

PYLEARN_DATA_PATH = os.environ["PYLEARN2_DATA_PATH"]


def preprocess_uwash_depth_dataset(attribs):

    pipeline = preprocessing.Pipeline()

    # rgbd : (num_examples, 72, 72, 4)
    # labels: (num_examples, 1)
    pipeline.items.append(hdf5_data_preprocessors.ExtractRawUWashData(attribs["raw_data_folder"],
                                                                      data_labels=("rgbd_patches", "patch_labels")))

    pipeline.items.append(hdf5_data_preprocessors.PerChannelGlobalContrastNormalizePatches(
        data_to_normalize_key='rgbd_patches',
        normalized_data_key='normalized_rgbd_patches',
        batch_size=100))

    #this extracts a valid set and test set
    pipeline.items.append(hdf5_data_preprocessors.SplitData(
        data_to_split_key=('rgbd_patches', 'patch_labels'),
        sets=attribs["sets"],
        patch_shape=attribs["patch_shape"],
        num_patches_per_set=attribs["num_patches_per_set"]))

    #now lets actually make a new dataset and run it through the pipeline
    hd5f_dataset = h5py.File(attribs["output_filepath"])
    pipeline.apply(hd5f_dataset)


if __name__ == "__main__":

    preprocess_attribs = dict(sets=("test", "valid"),
                              num_patches_per_set=(10000, 10000),
                              patch_shape=(72, 72),
                              raw_data_folder=PYLEARN_DATA_PATH + "/rgbd-dataset",
                              output_filepath=PYLEARN_DATA_PATH + "/rgbd-dataset/rgbd_preprocessed_72x72.h5")

    preprocess_uwash_depth_dataset(preprocess_attribs)
