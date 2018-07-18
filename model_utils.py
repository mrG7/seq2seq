from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
# from helper_functions import *
from resnet_model import ResNet, get_block_sizes
from tensorflow.contrib.slim.nets import resnet_v2
import numpy as np

# import six
# from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
# from tensorflow.python.layers import utils
# from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
# from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops


def frontend_3D(x_input, training=True, name="3d_conv"):
    # with tf.name_scope(name):
    # BATCH_SIZE, NUM_FRAMES, HEIGHT, WIDTH, NUM_CHANNELS = x_input.get_shape().as_list()
    # NUM_CLASSES = 500

    # 3D CONVOLUTION
    # First Convolution Layer - Image input, use 64 3D filters of size 5x5x5
    # shape of weights is (dx, dy, dz, #in_filters, #out_filters)
    n = np.prod([5, 7, 7, 64]) # std for weight initialization
    W_conv1 = tf.get_variable(name="W", initializer=tf.truncated_normal(shape=[5, 7, 7, 1, 64], stddev=np.sqrt(2/n)))
    b_conv1 = tf.get_variable(name="b", initializer=tf.constant(0.1, shape=[64]))
    # apply first convolution
    z_conv1 = tf.nn.conv3d(x_input, W_conv1, strides=[1, 1, 2, 2, 1], padding='SAME') + b_conv1
    # apply batch normalization
    z_conv1_bn = tf.contrib.layers.batch_norm(z_conv1,
                                              data_format='NHWC',  # Matching the "cnn" tensor shape
                                              center=True,
                                              scale=True,
                                              is_training=training,
                                              scope='cnn3d-batch_norm')
    # apply relu activation
    h_conv1 = tf.nn.relu(z_conv1_bn)
    print("shape after 1st convolution is %s" % h_conv1.get_shape)

    # apply max pooling
    h_pool1 = tf.nn.max_pool3d(h_conv1,
                               strides=[1, 1, 2, 2, 1],
                               ksize=[1, 3, 3, 3, 1],
                               padding='SAME')
    print(h_pool1.get_shape)
    print("shape after 1st pooling is %s" % h_pool1.get_shape)
    return h_pool1


def conv_backend(inputs, options):
    shape = inputs.get_shape().as_list()
    print('Temporal convolution backend')
    print('input shape %s' % shape)
    inputs = tf.layers.conv1d(inputs=inputs, filters=2*shape[-1], kernel_size=5, strides=2,
                              padding='valid',  # 'same'
                              data_format='channels_last', dilation_rate=1, activation=None,
                              use_bias=True, kernel_initializer=None, bias_initializer=tf.zeros_initializer(),
                              kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                              kernel_constraint=None, bias_constraint=None, trainable=True,
                              name=None, reuse=None)
    # print(inputs.get_shape())
    inputs = tf.layers.batch_normalization(inputs)
    inputs = tf.nn.relu(inputs)
    inputs = tf.layers.max_pooling1d(inputs=inputs, pool_size=2, strides=2, padding='valid',
                                     data_format='channels_last', name=None)
    inputs = tf.layers.conv1d(inputs=inputs, filters=4 * shape[-1], kernel_size=5, strides=2,
                              padding='valid',  # 'same'
                              data_format='channels_last', dilation_rate=1, activation=None,
                              use_bias=True, kernel_initializer=None, bias_initializer=tf.zeros_initializer(),
                              kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                              kernel_constraint=None, bias_constraint=None, trainable=True,
                              name=None, reuse=None)
    inputs = tf.layers.batch_normalization(inputs)
    inputs = tf.nn.relu(inputs)
    inputs = tf.reduce_mean(inputs, axis=1)
    # print(inputs.get_shape())
    inputs = tf.layers.dense(inputs=inputs, units=shape[-1], activation=None, use_bias=True,
                             kernel_initializer=None, bias_initializer=tf.zeros_initializer(),
                             kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                             kernel_constraint=None, bias_constraint=None, trainable=True,
                             name=None, reuse=None)
    inputs = tf.layers.batch_normalization(inputs)
    inputs = tf.nn.relu(inputs)
    inputs = tf.layers.dense(inputs=inputs, units=options['num_classes'], activation=None, use_bias=True,
                             kernel_initializer=None, bias_initializer=tf.zeros_initializer(),
                             kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                             kernel_constraint=None, bias_constraint=None, trainable=True,
                             name=None, reuse=None)
    return inputs


def backend_resnet(x_input, resnet_size=34, final_size=512, num_classes=None, frontend_3d=False, training=True, name="resnet"):

    with tf.name_scope(name):
        BATCH_SIZE, NUM_FRAMES, HEIGHT, WIDTH, NUM_CHANNELS = x_input.get_shape().as_list()

        # RESNET
        video_input = tf.reshape(x_input, (-1, HEIGHT, WIDTH, NUM_CHANNELS))  # BATCH_SIZE*NUM_FRAMES

        #  = tf.cast(video_input, tf.float32)
        resnet = ResNet(resnet_size=resnet_size, bottleneck=False, num_classes=num_classes, num_filters=64,
                        kernel_size=7, conv_stride=2, first_pool_size=3, first_pool_stride=2,
                        second_pool_size=7, second_pool_stride=1, block_sizes=get_block_sizes(resnet_size),
                        block_strides=[1, 2, 2, 2], final_size=final_size, frontend_3d=frontend_3d)
        features = resnet.__call__(video_input, training=training)
        # features, end_points = resnet_v2.resnet_v1_50(video_input, None)
        features = tf.reshape(features, (BATCH_SIZE, -1, int(features.get_shape()[1])))  # NUM_FRAMES

        print("shape after resnet is %s" % features.get_shape())

    return features


def backend_resnet50_v2_slim(x_input):
    BATCH_SIZE, NUM_FRAMES, HEIGHT, WIDTH, NUM_CHANNELS = x_input.get_shape().as_list()

    # RESNET
    video_input = tf.reshape(x_input, (BATCH_SIZE * NUM_FRAMES, HEIGHT, WIDTH, NUM_CHANNELS))

    features, end_points = resnet_v2.resnet_v2_50(video_input, num_classes=512)
    # features, end_points = resnet_v2.resnet_v1_50(video_input, None)
    features = tf.reshape(features, (BATCH_SIZE, NUM_FRAMES, int(features.get_shape()[3])))

    print("shape after resnet is %s" % features.get_shape())

    return features


def concat_resnet_output(resnet_out):
    # add resnet outputs per frame. This vector estimates resnets prediction on the word spoken
    return tf.reduce_mean(resnet_out, axis=1)


def serialize_resnet_output(resnet_out):
    BATCH_SIZE, NUM_FRAMES, NUM_CLASSES = resnet_out.get_shape()
    return tf.reshape(resnet_out, (BATCH_SIZE, NUM_FRAMES*NUM_CLASSES))


def fully_connected_logits(x_input, out_size, name='fc_logits'):
    with tf.name_scope(name):
        # fully connected layer (512 -> 500)
        # predictions = tf.layers.dropout(inputs=x_input, rate=0.5)
        predictions = tf.layers.dense(inputs=x_input, units=out_size, activation=tf.nn.relu)
        # No Softmax here as this will be accounted for by the loss function
        # predictions = tf.nn.softmax(predictions)
        print("shape of predictions is %s" % predictions.get_shape())
        return predictions


def stacked_lstm(input_forw, num_layers, num_hidden, residual=False, use_peepholes=True, return_cell=False):
    """
    input_forw : (tensor) input tensor forward in time
    num_layers : (int) depth of stacked LSTM
    num_hidden : (int, list, tuple) number of units at each LSTM layer
    """
    if type(num_hidden) is int:
        num_hidden = [num_hidden] * num_layers
    # print(len(num_hidden))
    # assert len(num_hidden) == num_layers \
    #     "length of num_hidden %d, must match num_layers %d" % (len(num_hidden), num_layers)
    # with tf.name_scope(name):
    # input_back = tf.reverse(input_forw, axis=[1])
    if residual:
        rnn_layers = [tf.contrib.rnn.ResidualWrapper(
                      tf.contrib.rnn.LSTMCell(
                         num_units=layer_size_,
                         use_peepholes=use_peepholes, cell_clip=None, initializer=None, num_proj=None, proj_clip=None,
                         forget_bias=1.0, state_is_tuple=True,
                         activation=tf.tanh, reuse=None, name=None))
                      for _, layer_size_ in enumerate(num_hidden)]
    else:
        rnn_layers = [tf.contrib.rnn.LSTMCell(
                         num_units=layer_size_,
                         use_peepholes=use_peepholes, cell_clip=None, initializer=None, num_proj=None, proj_clip=None,
                         forget_bias=1.0, state_is_tuple=True,
                         activation=tf.tanh, reuse=None, name=None)
                      for _, layer_size_ in enumerate(num_hidden)]
    # create a RNN cell composed sequentially of a number of RNNCells
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
    if return_cell:
        return multi_rnn_cell
    outputs, states = tf.nn.dynamic_rnn(multi_rnn_cell, input_forw, dtype=tf.float32)
    outputs = tf.concat(outputs, 2)
    return outputs, states


def blstm_encoder(input_forw, model_options):
    """
    input_forw : input tensor forward in time
    """
    if 'encoder_dropout_keep_prob' in model_options:
        dropout_keep_prob = model_options['encoder_dropout_keep_prob']
    else:
        dropout_keep_prob = 1.0

        # with tf.name_scope(name):
    # input_back = tf.reverse(input_forw, axis=[1])
    if model_options['residual_encoder']:
        print('encoder : residual BLSTM')
        rnn_layers = [tf.contrib.rnn.ResidualWrapper(
                      tf.contrib.rnn.LayerNormBasicLSTMCell(model_options['encoder_num_hidden'],
                                                            forget_bias=1.0,
                                                            input_size=None,
                                                            activation=tf.tanh,
                                                            layer_norm=True,
                                                            norm_gain=1.0,
                                                            norm_shift=0.0,
                                                            dropout_keep_prob=dropout_keep_prob,
                                                            dropout_prob_seed=None,
                                                            reuse=None))
                      for _ in range(model_options['encoder_num_layers'])]
    else:
        print('encoder : BLSTM')
        rnn_layers = [tf.contrib.rnn.LayerNormBasicLSTMCell(model_options['encoder_num_hidden'],
                                                            forget_bias=1.0,
                                                            input_size=None,
                                                            activation=tf.tanh,
                                                            layer_norm=True,
                                                            norm_gain=1.0,
                                                            norm_shift=0.0,
                                                            dropout_keep_prob=dropout_keep_prob,
                                                            dropout_prob_seed=None,
                                                            reuse=None)
                      for _ in range(model_options['encoder_num_layers'])]
    # create a RNN cell composed sequentially of a number of RNNCells
    multi_rnn_cell_forw = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
    multi_rnn_cell_back = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

    outputs, hidden_states = tf.nn.bidirectional_dynamic_rnn(
                                multi_rnn_cell_forw, multi_rnn_cell_back,
                                input_forw, dtype=tf.float32)
    outputs = tf.concat(outputs, 2)
    # hidden_states = tf.concat(hidden_states, 3)
    # hidden_states_list = []
    # for layer_id in range(model_options['encoder_num_layers']):
    #     hidden_states_list.append(hidden_states[0][layer_id])  # forward
    #     hidden_states_list.append(hidden_states[1][layer_id])  # backward
    # hidden_states = tuple(hidden_states_list)
    return outputs, hidden_states

    # 'outputs' is a tensor of shape [batch_size, max_time, 256]
    # 'state' is a N-tuple where N is the number of LSTMCells containing a
    # tf.contrib.rnn.LSTMStateTuple for each cell
    # outputs_forw, _ = tf.nn.dynamic_rnn(cell=multi_rnn_cell_forw,
    #                                     inputs=input_forw,
    #                                     dtype=tf.float32)
    # outputs_back, _ = tf.nn.dynamic_rnn(cell=multi_rnn_cell_back,
    #                                     inputs=input_back,
    #                                     dtype=tf.float32)
    # bilstm_out = tf.concat([last_forw, last_back], axis=1)
    # print("shape of bilstm output is %s" % bilstm_out.get_shape())
    # return bilstm_out

def blstm_2layer(x_input, name="blstm_2layer"):
    with tf.name_scope(name):
        # 2-layer BiLSTM
        # dense_out = tf.concat(dense_out, axis=1)
        # Define input for forward and backward LSTM
        dense_out_forw = x_input #tf.squeeze(dense_out, axis=2)
        dense_out_back = tf.reverse(dense_out_forw, axis=[1])
        # create 2 layer LSTMCells
        # rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [256, 256]]
        rnn_layers = [tf.contrib.rnn.LayerNormBasicLSTMCell(size) for size in [256, 256]]

        # create a RNN cell composed sequentially of a number of RNNCells
        multi_rnn_cell_forw = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
        multi_rnn_cell_back = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

        # 'outputs' is a tensor of shape [batch_size, max_time, 256]
        # 'state' is a N-tuple where N is the number of LSTMCells containing a
        # tf.contrib.rnn.LSTMStateTuple for each cell
        outputs_forw, _ = tf.nn.dynamic_rnn(cell=multi_rnn_cell_forw,
                                            inputs=dense_out_forw,
                                            dtype=tf.float32)
        outputs_back, _ = tf.nn.dynamic_rnn(cell=multi_rnn_cell_back,
                                            inputs=dense_out_back,
                                            dtype=tf.float32)

        # get only the last output from lstm
        # lstm_out = tf.transpose(lstm_out, [1, 0, 2])
        last_forw = tf.gather(outputs_forw, indices=int(outputs_forw.get_shape()[1]) - 1, axis=1)
        last_back = tf.gather(outputs_back, indices=int(outputs_forw.get_shape()[1]) - 1, axis=1)

        bilstm_out = tf.concat([last_forw, last_back], axis=1)
        print("shape of bilstm output is %s" % bilstm_out.get_shape())
        return bilstm_out

def lstm_1layer(x_input, size=256, name="lstm_1layer"):
    with tf.name_scope(name):
        # 2-layer BiLSTM
        # dense_out = tf.concat(dense_out, axis=1)
        # Define input for forward and backward LSTM
        #dense_out_forw = x_input #tf.squeeze(dense_out, axis=2)
        #dense_out_back = tf.reverse(dense_out_forw, axis=[1])
        # create 2 layer LSTMCells
        # rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [256, 256]]
        rnn_layers = tf.contrib.rnn.LSTMCell(size)

        # create a RNN cell composed sequentially of a number of RNNCells
        multi_rnn_cell_forw = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
        #multi_rnn_cell_back = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

        # 'outputs' is a tensor of shape [batch_size, max_time, 256]
        # 'state' is a N-tuple where N is the number of LSTMCells containing a
        # tf.contrib.rnn.LSTMStateTuple for each cell
        lstm_out, _ = tf.nn.dynamic_rnn(cell=multi_rnn_cell_forw,
                                            inputs=x_input,
                                            dtype=tf.float32)
        # outputs_back, _ = tf.nn.dynamic_rnn(cell=multi_rnn_cell_back,
        #                                     inputs=dense_out_back,
        #                                     dtype=tf.float32)

        # get only the last output from lstm
        # lstm_out = tf.transpose(lstm_out, [1, 0, 2])
        #lstm_out = tf.gather(outputs_forw, indices=int(outputs_forw.get_shape()[1]) - 1, axis=1)
        #last_back = tf.gather(outputs_back, indices=int(outputs_forw.get_shape()[1]) - 1, axis=1)

        #bilstm_out = tf.concat([last_forw, last_back], axis=1)
        print("shape of lstm output is %s" % lstm_out.get_shape())
        return lstm_out



def get_model_variables():
    vars = [v for v in tf.trainable_variables() if "batch_normalization" not in v.name
                                                             and "gamma" not in v.name
                                                              and "beta" not in v.name]
    vars = [v for v in tf.global_variables()+tf.local_variables() if "batch_normalization" not in v.name
                                                             and "gamma" not in v.name
                                                              and "beta" not in v.name
                                                              and "Adam" not in v.name]
    return vars


class MultiLayerOutput(base.Layer):
    """2x Densely-connected layers class.
    Implements the operation:
    `outputs = activation(inputs * kernel + bias)`
    twice
    Where `activation` is the activation function passed as the `activation`
    argument (if not `None`), `kernel` is a weights matrix created by the layer,
    and `bias` is a bias vector created by the layer
    (only if `use_bias` is `True`).
    Note: if the input to the layer has a rank greater than 2, then it is
    flattened prior to the initial matrix multiply by `kernel`.
    Arguments:
    units: Integer or Long, dimensionality of the output space.
    activation: Activation function (callable). Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: Initializer function for the weight matrix.
      If `None` (default), weights are initialized using the default
      initializer used by `tf.get_variable`.
    bias_initializer: Initializer function for the bias.
    kernel_regularizer: Regularizer function for the weight matrix.
    bias_regularizer: Regularizer function for the bias.
    activity_regularizer: Regularizer function for the output.
    kernel_constraint: An optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    bias_constraint: An optional projection function to be applied to the
        bias after being updated by an `Optimizer`.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such cases.
    reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.
    Properties:
    units: Python integer, dimensionality of the output space.
    activation: Activation function (callable).
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: Initializer instance (or name) for the kernel matrix.
    bias_initializer: Initializer instance (or name) for the bias.
    kernel_regularizer: Regularizer instance for the kernel matrix (callable)
    bias_regularizer: Regularizer instance for the bias (callable).
    activity_regularizer: Regularizer instance for the output (callable)
    kernel_constraint: Constraint function for the kernel matrix.
    bias_constraint: Constraint function for the bias.
    kernel: Weight matrix (TensorFlow variable or tensor).
    bias: Bias vector, if applicable (TensorFlow variable or tensor).
    """

    def __init__(self, units,
               activation=[tf.nn.relu, None],
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=init_ops.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
        super(MultiLayerOutput, self).__init__(trainable=trainable, name=name,
                                    activity_regularizer=activity_regularizer,
                                    **kwargs)
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.input_spec = base.InputSpec(min_ndim=2)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                           'should be defined. Found `None`.')
        self.input_spec = base.InputSpec(min_ndim=2,
                                         axes={-1: input_shape[-1].value})
        self.kernel1 = self.add_variable('kernel1',
                                        shape=[input_shape[-1].value, self.units[0]],
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        dtype=self.dtype,
                                        trainable=True)
        self.kernel2 = self.add_variable('kernel2',
                                        shape=[self.units[0], self.units[1]],
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        dtype=self.dtype,
                                        trainable=True)
        self.bias1 = self.add_variable('bias1',
                                    shape=[self.units[0],],
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint,
                                    dtype=self.dtype,
                                    trainable=True) if self.use_bias else None
        self.bias2 = self.add_variable('bias2',
                                     shape=[self.units[1], ],
                                     initializer=self.bias_initializer,
                                     regularizer=self.bias_regularizer,
                                     constraint=self.bias_constraint,
                                     dtype=self.dtype,
                                     trainable=True) if self.use_bias else None

    def call(self, inputs):
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        shape = inputs.get_shape().as_list()
        if len(shape) > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(inputs, self.kernel1, [[len(shape) - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if context.in_graph_mode():
                output_shape = shape[:-1] + [self.units[0]]
                outputs.set_shape(output_shape)
            if self.use_bias:
                outputs = nn.bias_add(outputs, self.bias)
            outputs = tf.layers.batch_normalization(outputs)
            outputs = self.activation[0](outputs)
            outputs = standard_ops.tensordot(outputs, self.kernel2, [[len(shape) - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if context.in_graph_mode():
                output_shape = shape[:-1] + [self.units[1]]
                outputs.set_shape(output_shape)
            if self.use_bias:
                outputs = nn.bias_add(outputs, self.bias2)
            if self.activation[1]:
                outputs = self.activation[1](outputs)
        else:
            outputs = standard_ops.matmul(inputs, self.kernel1)
            if self.use_bias:
                outputs = nn.bias_add(outputs, self.bias1)
            outputs = tf.layers.batch_normalization(outputs)
            outputs = self.activation[0](outputs)
            outputs = standard_ops.matmul(outputs, self.kernel2)
            if self.use_bias:
                outputs = nn.bias_add(outputs, self.bias2)
            if self.activation[1]:
                outputs = self.activation[1](outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if input_shape[-1].value is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        return input_shape[:-1].concatenate(self.units[1])