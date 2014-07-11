import cPickle
import numpy as np
import theano
import theano.tensor as T

#
# def get_pixel_classifier():
#     f = open('tests/mlp_3_best.pkl')
#     mlp_model = cPickle.load(f)
#
#     X = mlp_model.get_input_space().make_theano_batch()
#     Y = mlp_model.fprop(X)
#     Y = T.argmax(Y, axis=1)
#     f = theano.function([X], Y)
#     return f
#
# def get_input_batches():
#     f = open('cnn_rgbd_preprocessed_train_dataset.pkl')
#     dataset = cPickle.load(f)
#     image = dataset.X[0:480/8*640/8]
#     return image
#
#
# f = get_pixel_classifier()
# input_batches = get_input_batches()
# for index in range(input_batches.shape[0]):
#     test_input = np.expand_dims(np.float32(input_batches[index]), 0)
#     output = f(test_input)
#     print output
# test_input = np.float32(np.zeros((1, 256)))
# output = f(test_input)
#
# print output
