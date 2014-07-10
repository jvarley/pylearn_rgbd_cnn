from pylearn2.datasets import preprocessing
import copy, logging, time, warnings, os, numpy, scipy


class ExtractPatches(preprocessing.Preprocessor):
    """
    Converts an image dataset into a dataset of patches
    extracted at random from the original dataset.

    Parameters
    ----------
    patch_shape : WRITEME
    num_patches : WRITEME
    rng : WRITEME
    """

    def __init__(self, patch_shape, num_patches, rng=None):
        self.patch_shape = patch_shape
        self.num_patches = num_patches

        self.start_rng = preprocessing.make_np_rng(copy.copy(rng), [1,2,3], which_method="randint")

    def apply(self, dataset, can_fit=False):
        """
        dataset : rgbd_dataset with axes (b, 0, 1, c)
        """
        rng = copy.copy(self.start_rng)

        X = dataset.get_topological_view()
        y = dataset.y

        num_topological_dimensions = len(X.shape) - 2

        if num_topological_dimensions != len(self.patch_shape):
            raise ValueError("ExtractPatches with "
                             + str(len(self.patch_shape))
                             + " topological dimensions called on "
                             + "dataset with "
                             + str(num_topological_dimensions) + ".")

        # batch size
        output_shape = [self.num_patches]
        # topological dimensions
        for dim in self.patch_shape:
            output_shape.append(dim)
        # number of channels
        output_shape.append(X.shape[-1])
        output_topo_x = numpy.zeros(output_shape, dtype=X.dtype)
        output_y = numpy.zeros(self.num_patches)
        channel_slice = slice(0, X.shape[-1])
        for i in xrange(self.num_patches):

            image_num = rng.randint(X.shape[0])
            x_args = [image_num]
            y_args = [image_num]

            for j in xrange(num_topological_dimensions):
                max_coord = X.shape[j + 1] - self.patch_shape[j]
                coord = rng.randint(max_coord + 1)
                x_args.append(slice(coord, coord + self.patch_shape[j]))
                y_args.append(coord + self.patch_shape[j]/2.0)
            x_args.append(channel_slice)
            output_topo_x[i, :] = X[x_args]
            output_y[i] = y[tuple(y_args)]

        dataset.set_topological_view(output_topo_x)
        dataset.y = output_y

        if output_y.shape[0] != self.num_patches and len(output_y.shape) == 1:
            raise ValueError("output_y has wrong shape!, should be: " +
                             str(self.num_patches) + " but is: " +
                             str(output_y.shape))

        if output_topo_x.shape != (self.num_patches, self.patch_shape[0], self.patch_shape[1], 4):
            raise ValueError("output_topo_x has wrong shape!, should be: (" +
                             str(self.num_patches) + ", " + str(self.patch_shape) + "," + str(4) + " but is: " +
                             str(output_topo_x.shape))

