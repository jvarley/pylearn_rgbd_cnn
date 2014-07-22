
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

    # #add the steps necessary to generate data for
    # # valid, test and training datasets
    # for i in range(len(attribs["sets"])):
    #
    #     which_set = attribs["sets"][i]
    #     num_patches = attribs["num_patches_per_set"][i]
    #
    #     #labels for the hdf5 file
    #     patch_label = which_set + "_patches"
    #     flattened_patch_label = which_set + "_flattened_patches"
    #     patch_labels = (patch_label, which_set + "_patch_labels")
    #
    #     pipeline.items.append(hdf5_data_preprocessors.ExtractPatches(patch_shape=attribs["patch_shape"],
    #                                                                  patch_labels=patch_labels,
    #                                                                  patch_source_labels=("rgbd", "labels"),
    #                                                                  num_patches=num_patches))
    #
    #     pipeline.items.append(hdf5_data_preprocessors.FlattenPatches(patch_label=patch_label,
    #                                                                  flattened_patch_label=flattened_patch_label))
    #
    #     pipeline.items.append(hdf5_data_preprocessors.GlobalContrastNormalizePatches(
    #         data_to_normalize_key=flattened_patch_label,
    #         batch_size=100,
    #         subtract_mean=True,
    #         scale=1.,
    #         sqrt_bias=0.,
    #         use_std=False,
    #         min_divisor=1e-8))

    #now lets actually make a new dataset and run it through the pipeline
    hd5f_dataset = h5py.File(attribs["output_filepath"])
    pipeline.apply(hd5f_dataset)


if __name__ == "__main__":

    preprocess_attribs = dict(sets=("train", "test", "valid"),
                              num_patches_per_set=(100000, 10000, 10000),
                              patch_shape=(72, 72),
                              raw_data_folder=PYLEARN_DATA_PATH + "/rgbd-dataset",
                              output_filepath=PYLEARN_DATA_PATH + "/rgbd-dataset/rgbd_preprocessed_72x72.h5")

    preprocess_uwash_depth_dataset(preprocess_attribs)
