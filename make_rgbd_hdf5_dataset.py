
import h5py
import os

from pylearn2.datasets import preprocessing

import hdf5_data_preprocessor

PYLEARN_DATA_PATH = os.environ["PYLEARN2_DATA_PATH"]


def preprocess_nyu_depth_dataset(attribs):

    pipeline = preprocessing.Pipeline()

    # rgbd : (1449, 640, 480, 4)
    # labels: (1449, 640, 480)
    pipeline.items.append(hdf5_data_preprocessor.ExtractRawNYUData(attribs["raw_filepath"], data_labels=("rgbd", "labels")))

    #add the steps necessary to generate data for
    # valid, test and training datasets
    for i in range(len(attribs["sets"])):

        which_set = attribs["sets"][i]
        num_patches = attribs["num_patches_per_set"][i]

        #labels for the hdf5 file
        patch_label = which_set + "_patches"
        flattened_patch_label = which_set + "_flattened_patches"
        patch_labels = (patch_label, which_set + "_patch_labels")

        pipeline.items.append(hdf5_data_preprocessor.ExtractPatches(patch_shape=attribs["patch_shape"],
                                                                    patch_labels=patch_labels,
                                                                    patch_source_labels=("rgbd", "labels"),
                                                                    num_patches=num_patches))

        pipeline.items.append(hdf5_data_preprocessor.FlattenPatches(patch_label=patch_label,
                                                                    flattened_patch_label=flattened_patch_label))

        pipeline.items.append(hdf5_data_preprocessor.GlobalContrastNormalizePatches(
            data_to_normalize_key=flattened_patch_label,
            batch_size=100,
            subtract_mean=True,
            scale=1.,
            sqrt_bias=0.,
            use_std=False,
            min_divisor=1e-8))

    #now lets actually make a new dataset and run it through the pipeline
    hd5f_dataset = h5py.File(attribs["output_filepath"])
    pipeline.apply(hd5f_dataset)


if __name__ == "__main__":

    preprocess_attribs = {

        "sets": ("train", "test", "valid"),
        "num_patches_per_set": (100000, 10000, 10000),

        "patch_shape": (25, 25),

        "raw_filepath": PYLEARN_DATA_PATH + "/nyu_depth_labeled/nyu_depth_v2_labeled.mat",
        "output_filepath": PYLEARN_DATA_PATH + "/nyu_depth_labeled/rgbd_preprocessed.h5"
    }

    preprocess_nyu_depth_dataset(preprocess_attribs)
