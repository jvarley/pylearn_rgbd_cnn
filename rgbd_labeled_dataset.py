
import h5py
import cPickle
import warnings
import functools
import numpy as np


from pylearn2.utils.iteration import (
    resolve_iterator_class
)

from pylearn2.datasets import dense_design_matrix
from pylearn2.datasets.dataset import Dataset
from pylearn2.utils import safe_zip, iteration,  safe_izip, wraps

import pylearn2.space
import pylearn2.utils.one_hot
from pylearn2.space import CompositeSpace
from pylearn2.utils.data_specs import is_flat_specs


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

            return labels

        data = h5py.File(datafile, "r")
        num_images = 1449
        if which_set == "train":
            samples_range = (0, num_images)
        else:
            samples_range = (0, num_images)

        topo_x = load_rgbd_data(data, samples_range)
        topo_y = load_labels(data, samples_range)

        super(RGBDLabeledDataset, self).__init__(topo_view=topo_x, y=topo_y)


    #have to override this function
    #the only change is the type of iterator
    #returned on the very last line
    @functools.wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=None, rng=None, data_specs=None,
                 return_tuple=False):

        if topo is not None or targets is not None:
            if data_specs is not None:
                raise ValueError('In DenseDesignMatrix.iterator, both the '
                                 '"data_specs" argument and deprecated '
                                 'arguments "topo" or "targets" were '
                                 'provided.',
                                 (data_specs, topo, targets))

            warnings.warn("Usage of `topo` and `target` arguments are "
                          "being deprecated, and will be removed "
                          "around November 7th, 2013. `data_specs` "
                          "should be used instead.",
                          stacklevel=2)

            # build data_specs from topo and targets if needed
            if topo is None:
                topo = getattr(self, '_iter_topo', False)
            if topo:
                # self.iterator is called without a data_specs, and with
                # "topo=True", so we use the default topological space
                # stored in self.X_topo_space
                assert self.X_topo_space is not None
                X_space = self.X_topo_space
            else:
                X_space = self.X_space

            if targets is None:
                targets = getattr(self, '_iter_targets', False)
            if targets:
                assert self.y is not None
                y_space = self.data_specs[0].components[1]
                space = CompositeSpace((X_space, y_space))
                source = ('features', 'targets')
            else:
                space = X_space
                source = 'features'

            data_specs = (space, source)
            convert = None

        else:
            if data_specs is None:
                data_specs = self._iter_data_specs

            # If there is a view_converter, we have to use it to convert
            # the stored data for "features" into one that the iterator
            # can return.
            space, source = data_specs
            if isinstance(space, CompositeSpace):
                sub_spaces = space.components
                sub_sources = source
            else:
                sub_spaces = (space,)
                sub_sources = (source,)

            convert = []
            for sp, src in safe_zip(sub_spaces, sub_sources):
                if src == 'features' and \
                   getattr(self, 'view_converter', None) is not None:
                    conv_fn = (lambda batch, self=self, space=sp:
                               self.view_converter.get_formatted_batch(batch,
                                                                       space))
                else:
                    conv_fn = None

                convert.append(conv_fn)

        # TODO: Refactor
        if mode is None:
            if hasattr(self, '_iter_subset_class'):
                mode = self._iter_subset_class
            else:
                raise ValueError('iteration mode not provided and no default '
                                 'mode set for %s' % str(self))
        else:
            mode = resolve_iterator_class(mode)

        if batch_size is None:
            batch_size = getattr(self, '_iter_batch_size', None)
        if num_batches is None:
            num_batches = getattr(self, '_iter_num_batches', None)
        if rng is None and mode.stochastic:
            rng = self.rng
        return FiniteDatasetExpandingYIterator(self,
                                     mode(self.X.shape[0],
                                          batch_size,
                                          num_batches,
                                          rng),
                                     data_specs=data_specs,
                                     return_tuple=return_tuple,
                                     convert=convert)


#this just allows us to store y where the number of categories
#is extremely large without have to create a vector
#with 1000 entries for a single 1.
class FiniteDatasetExpandingYIterator(object):
    """
    A wrapper around subset iterators that actually retrieves
    data.

    Parameters
    ----------
    dataset : `Dataset` object
        The dataset over which to iterate.
    data_specs : tuple
        A `(space, source)` tuple. See :ref:`data_specs` for a full
        description. Must not contain nested composite spaces.
    subset_iterator : object
        An iterator object that returns slice objects or lists of
        examples, conforming to the interface specified by
        :py:class:`SubsetIterator`.
    return_tuple : bool, optional
        Always return a tuple, even if there is exactly one source
        of data being returned. Defaults to `False`.
    convert : list of callables
        A list of callables, in the same order as the sources
        in `data_specs`, that will be called on the individual
        source batches prior to any further processing.

    Notes
    -----
    See the documentation for :py:class:`SubsetIterator` for
    attribute documentation.
    """

    def __init__(self, dataset, subset_iterator, data_specs=None,
                 return_tuple=False, convert=None):
        self._data_specs = data_specs
        self._dataset = dataset
        self._subset_iterator = subset_iterator
        self._return_tuple = return_tuple

        # Keep only the needed sources in self._raw_data.
        # Remember what source they correspond to in self._source
        assert is_flat_specs(data_specs)

        dataset_space, dataset_source = self._dataset.get_data_specs()
        assert is_flat_specs((dataset_space, dataset_source))

        # the dataset's data spec is either a single (space, source) pair,
        # or a pair of (non-nested CompositeSpace, non-nested tuple).
        # We could build a mapping and call flatten(..., return_tuple=True)
        # but simply putting spaces, sources and data in tuples is simpler.
        if not isinstance(dataset_source, tuple):
            dataset_source = (dataset_source,)

        if not isinstance(dataset_space, CompositeSpace):
            dataset_sub_spaces = (dataset_space,)
        else:
            dataset_sub_spaces = dataset_space.components
        assert len(dataset_source) == len(dataset_sub_spaces)

        all_data = self._dataset.get_data()
        if not isinstance(all_data, tuple):
            all_data = (all_data,)

        space, source = data_specs
        if not isinstance(source, tuple):
            source = (source,)
        if not isinstance(space, CompositeSpace):
            sub_spaces = (space,)
        else:
            sub_spaces = space.components
        assert len(source) == len(sub_spaces)

        self._raw_data = tuple(all_data[dataset_source.index(s)]
                               for s in source)
        self._source = source

        if convert is None:
            self._convert = [None for s in source]
        else:
            assert len(convert) == len(source)
            self._convert = convert

        for i, (so, sp, dt) in enumerate(safe_izip(source,
                                                   sub_spaces,
                                                   self._raw_data)):
            idx = dataset_source.index(so)
            dspace = dataset_sub_spaces[idx]

            init_fn = self._convert[i]
            fn = init_fn

            # If there is an init_fn, it is supposed to take
            # care of the formatting, and it should be an error
            # if it does not. If there was no init_fn, then
            # the iterator will try to format using the generic
            # space-formatting functions.
            if init_fn is None:
                # "dspace" and "sp" have to be passed as parameters
                # to lambda, in order to capture their current value,
                # otherwise they would change in the next iteration
                # of the loop.
                if fn is None:
                    #######################################################
                    #this was added to to expand the y's just when we need them
                    dspace = pylearn2.space.VectorSpace(894)

                    def fn(batch, dspace=dspace, sp=sp):
                        try:
                            batch_2 = pylearn2.utils.one_hot.one_hot(batch.astype(int), max_label=893)
                            #return dspace.np_format_as(batch, sp)
                            return dspace.np_format_as(batch_2, sp)

                        except ValueError as e:
                            msg = str(e) + '\nMake sure that the model and '\
                                           'dataset have been initialized with '\
                                           'correct values.'
                            raise ValueError(msg)
                    #########################################################
                else:
                    fn = (lambda batch, dspace=dspace, sp=sp, fn_=fn:
                          dspace.np_format_as(fn_(batch), sp))

            self._convert[i] = fn

    def __iter__(self):
        return self

    @wraps(iteration.SubsetIterator.next)
    def next(self):
        """
        Retrieves the next batch of examples.

        Returns
        -------
        next_batch : object
            An object representing a mini-batch of data, conforming
            to the space specified in the `data_specs` constructor
            argument to this iterator. Will be a tuple if more
            than one data source was specified or if the constructor
            parameter `return_tuple` was `True`.

        Raises
        ------
        StopIteration
            When there are no more batches to return.
        """
        next_index = self._subset_iterator.next()
        # TODO: handle fancy-index copies by allocating a buffer and
        # using np.take()

        rval = tuple(
            fn(data[next_index]) if fn else data[next_index]
            for data, fn in safe_izip(self._raw_data, self._convert))
        if not self._return_tuple and len(rval) == 1:
            rval, = rval
        return rval

    @property
    @wraps(iteration.SubsetIterator.batch_size, assigned=(), updated=())
    def batch_size(self):
        return self._subset_iterator.batch_size

    @property
    @wraps(iteration.SubsetIterator.num_batches, assigned=(), updated=())
    def num_batches(self):
        return self._subset_iterator.num_batches

    @property
    @wraps(iteration.SubsetIterator.num_examples, assigned=(), updated=())
    def num_examples(self):
        return self._subset_iterator.num_examples

    @property
    @wraps(iteration.SubsetIterator.uneven, assigned=(), updated=())
    def uneven(self):
        return self._subset_iterator.uneven

    @property
    @wraps(iteration.SubsetIterator.stochastic, assigned=(), updated=())
    def stochastic(self):
        return self._subset_iterator.stochastic
