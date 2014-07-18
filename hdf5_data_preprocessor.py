
import h5py
import numpy as np

from pylearn2.datasets import preprocessing
from pylearn2.expr.preprocessing import global_contrast_normalize


#these borrow heavily from the pylearn2 preprocessing classes, just modified slightly
#to work with hdf5 and my data


class ExtractRawNYUData(preprocessing.Preprocessor):

    def __init__(self, raw_data_filepath,  data_labels=("rgbd", "labels")):
        self.raw_data_filepath = raw_data_filepath
        self.data_labels = data_labels

    def apply(self, dataset, can_fit=False):

        raw_dataset = h5py.File(self.raw_data_filepath)

        #check if we have already extracted the raw data
        if self.data_labels[0] in dataset.keys() or self.data_labels[1] in dataset.keys():
            print "skipping extract_raw_data, this has already been run"
            return

        num_images = raw_dataset["images"].shape[0]

        dataset.create_dataset(self.data_labels[0], (num_images, 640, 480, 4), chunks=(10, 640, 480, 4))
        dataset.create_dataset(self.data_labels[1], (num_images, 640, 480), chunks=(10, 640, 480))

        for i in range(num_images):
            if i % (num_images/10) == 0:
                print "extracting raw data: " + str(i) + "/" + str(num_images)

            dataset[self.data_labels[0]][i, :, :, 0:3] = np.rollaxis(raw_dataset["images"][i], 0, 3)
            dataset[self.data_labels[0]][i, :, :, 3] = raw_dataset["depths"][i]
            dataset[self.data_labels[1]][i] = raw_dataset["labels"][i]


class ExtractPatches(preprocessing.Preprocessor):

    def __init__(self,
                 patch_shape=(25, 25),
                 patch_labels=("rgbd_patches", "patch_labels"),
                 patch_source_labels=("rgbd", "labels"),
                 num_patches=100000):

        self.patch_shape = patch_shape
        self.patch_labels = patch_labels
        self.patch_source_labels = patch_source_labels
        self.num_patches = num_patches

    def apply(self, dataset, can_fit=False):
        #check if we have already extracted patches for this set of patch_labels
        if self.patch_labels[0] in dataset.keys() or self.patch_labels[1] in dataset.keys():
            print "skipping extract_patches, this has already been run"
            return

        #just rows and columns
        num_topological_dimensions = 2
        dataset.create_dataset(self.patch_labels[0], (self.num_patches, self.patch_shape[0], self.patch_shape[1], 4), chunks=(100, self.patch_shape[0], self.patch_shape[1], 4))
        dataset.create_dataset(self.patch_labels[1], (self.num_patches, 1))

        X = dataset[self.patch_source_labels[0]]
        y = dataset[self.patch_source_labels[1]]

        num_images = X.shape[0]

        patch_X = dataset[self.patch_labels[0]]
        patch_y = dataset[self.patch_labels[1]]

        channel_slice = slice(0, X.shape[-1])

        rng = preprocessing.make_np_rng([1, 2, 3], which_method="randint")

        for i in xrange(self.num_patches):
            if i % (self.num_patches/10) == 0:
                print "extracting patches: " + str(i) + "/" + str(self.num_patches)

            image_num = rng.randint(num_images)
            x_args = [image_num]
            y_args = [image_num]

            for j in xrange(num_topological_dimensions):
                max_coord = X.shape[j + 1] - self.patch_shape[j]
                coord = rng.randint(max_coord + 1)
                x_args.append(slice(coord, coord + self.patch_shape[j]))
                y_args.append(coord + self.patch_shape[j]/2.0)

            x_args.append(channel_slice)

            patch_X[i] = X[tuple(x_args)]
            patch_y[i] = y[tuple(y_args)]


class FlattenPatches(preprocessing.Preprocessor):

    def __init__(self,
                 patch_label="rgbd_patches",
                 flattened_patch_label="rgbd_flattened_patches"):
        self.patch_label = patch_label
        self.flattened_patch_label = flattened_patch_label

    def apply(self, dataset, can_fit=False):

        #check if we have already flattened patches
        if self.flattened_patch_label in dataset.keys():
            print "skipping flatten_patches, this has already been run"
            return
        else:
            print "flattening_patches"

        patches = dataset[self.patch_label]
        num_patches = patches.shape[0]
        num_flattened_features = 1
        for i in range(1, len(patches.shape)):
            num_flattened_features *= patches.shape[i]

        dataset.create_dataset(self.flattened_patch_label, (num_patches, num_flattened_features), chunks=(100, num_flattened_features))

        flattened_patches = dataset[self.flattened_patch_label]
        for i in range(num_patches):
            flattened_patches[i] = patches[i].flatten()


class GlobalContrastNormalizePatches(preprocessing.Preprocessor):

    def __init__(self,
                 data_to_normalize_key,
                 batch_size,
                 subtract_mean=True,
                 scale=1.,
                 sqrt_bias=0.,
                 use_std=False,
                 min_divisor=1e-8):

        self.data_to_normalize_key = data_to_normalize_key
        self.batch_size = batch_size
        self.subtract_mean = subtract_mean
        self.scale = scale
        self.sqrt_bias = sqrt_bias
        self.use_std = use_std
        self.min_divisor = min_divisor

    def apply(self, dataset, can_fit=False):

        data = dataset[self.data_to_normalize_key]
        data_size = data.shape[0]

        for i in xrange(0, data_size, self.batch_size):
            stop = i + self.batch_size

            X = data[i:stop]

            for index in range(i, stop):

                X_normalized = global_contrast_normalize(X,
                                                         scale=self.scale,
                                                         subtract_mean=self.subtract_mean,
                                                         use_std=self.use_std,
                                                         sqrt_bias=self.sqrt_bias,
                                                         min_divisor=self.min_divisor)
                data[i:stop] = X_normalized