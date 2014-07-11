import numpy as np
import h5py
import os

from pylearn2.datasets import preprocessing
from pylearn2.expr.preprocessing import global_contrast_normalize


def preprocess_h5py_dataset():

    pylearn_data_path = os.environ["PYLEARN2_DATA_PATH"]

    # this is the downloaded nyu data
    raw_hdf5_dataset_filepath = pylearn_data_path + "/nyu_depth_labeled/nyu_depth_v2_labeled.mat"

    # this is the preprocessed dataset save location
    hd5f_dataset_filepath = pylearn_data_path + "/nyu_depth_labeled/out.mat"
    hd5f_dataset = h5py.File(hd5f_dataset_filepath)

    # first pull the data out of the raw_datafile and format it
    # so that we can pass it into the HDF5Dataset constructor
    # rgbd : (1449, 640, 480, 4)
    # labels: (1449, 640, 480)
    extract_raw_data(raw_hdf5_dataset_filepath, hd5f_dataset_filepath)

    # Next we want to pull out small patches for training
    # rgbd_patches = (num_patches, patch_shape, 4)
    # patch_labels = (num_patches, 1)
    extract_patches(hd5f_dataset_filepath, patch_shape=(80,60), num_patches=100000)

    # Next we want to flatten patches as it just seems to be handled better,
    # and nont of the tools for viewing topological data are working well
    # for rgbd data
    # rgbd_flattened_patches : (num_patches, patch_shape[0]*patch_shape[1]*4)
    flatten_patches(hd5f_dataset)

    global_contrast_normalize_patch(hd5f_dataset,
                                    "rgbd_flattened_patches",
                                    batch_size=100,
                                    subtract_mean=True,
                                    scale=1.,
                                    sqrt_bias=0.,
                                    use_std=False,
                                    min_divisor=1e-8,
                                    )


def extract_raw_data(raw_datafilename, output_datafilename ):

        raw_data = h5py.File(raw_datafilename)
        output_data = h5py.File(output_datafilename)

        #check if we have already extracted the raw data
        if "rgbd" in output_data.keys():
            print "skipping extract_raw_data, this has already been run"
            return

        num_images = raw_data["images"].shape[0]

        output_data.create_dataset("rgbd", (num_images, 640, 480, 4), chunks=(10, 640, 480, 4))
        output_data.create_dataset("labels", (num_images, 640, 480), chunks=(10, 640, 480))

        for i in range(num_images):
            if i % (num_images/10) == 0:
                print "extracting raw data: " + str(i) + "/" + str(num_images)

            output_data["rgbd"][i, :, :, 0:3] = np.rollaxis(raw_data["images"][i], 0, 3)
            output_data["rgbd"][i, :, :, 3] = raw_data["depths"][i]
            output_data["labels"][i] = raw_data["labels"][i]


def extract_patches(hd5f_dataset_filepath, patch_shape, num_patches):

        h5py_file = h5py.File(hd5f_dataset_filepath)

        #check if we have already extracted patches
        if "rgbd_patches" in h5py_file.keys():
            print "skipping extract_patches, this has already been run"
            return

        #just rows and columns
        num_topological_dimensions = 2
        h5py_file.create_dataset("rgbd_patches", (num_patches, patch_shape[0], patch_shape[1], 4), chunks=(100, patch_shape[0], patch_shape[1], 4))
        h5py_file.create_dataset("patch_labels", (num_patches, 1))

        X = h5py_file["rgbd"]
        y = h5py_file["labels"]

        num_images = X.shape[0]

        patch_X = h5py_file["rgbd_patches"]
        patch_y = h5py_file["patch_labels"]

        channel_slice = slice(0, X.shape[-1])

        rng = preprocessing.make_np_rng([1, 2, 3], which_method="randint")

        for i in xrange(num_patches):
            if i % (num_patches/10) == 0:
                print "extracting patches: " + str(i) + "/" + str(num_patches)

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


def flatten_patches(hd5f_dataset):
    #check if we have already flattened patches
    if "rgbd_flattened_patches" in hd5f_dataset.keys():
        print "skipping flatten_patches, this has already been run"
        return
    else:
        print "flattening_patches"

    patches = hd5f_dataset["rgbd_patches"]
    num_patches = patches.shape[0]
    num_flattened_features = 1
    for i in range(1, len(patches.shape)):
        num_flattened_features *= patches.shape[i]

    hd5f_dataset.create_dataset("rgbd_flattened_patches", (num_patches, num_flattened_features), chunks=(100, num_flattened_features))

    flattened_patches = hd5f_dataset["rgbd_flattened_patches"]
    for i in range(num_patches):
        flattened_patches[i] = patches[i].flatten()


def global_contrast_normalize_patch(hd5f_dataset,
                                    data_to_normalize_key,
                                    batch_size,
                                    subtract_mean=True,
                                    scale=1.,
                                    sqrt_bias=0.,
                                    use_std=False,
                                    min_divisor=1e-8,
                                    ):

            data = hd5f_dataset[data_to_normalize_key]
            data_size = data.shape[0]

            for i in xrange(0, data_size, batch_size):
                stop = i + batch_size

                X = data[i:stop]

                for index in range(i, stop):

                    X_normalized = global_contrast_normalize(X,
                                                             scale=scale,
                                                             subtract_mean=subtract_mean,
                                                             use_std=use_std,
                                                             sqrt_bias=sqrt_bias,
                                                             min_divisor=min_divisor)
                    data[i:stop] = X_normalized


if __name__ == "__main__":
    preprocess_h5py_dataset()