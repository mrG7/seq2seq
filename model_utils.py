import tensorflow as tf
# from helper_functions import *
from resnet_model import ResNet, get_block_sizes
from tensorflow.contrib.slim.nets import resnet_v2
import numpy as np


def frontend_3D(x_input, training=True, name="3d_conv"):
    with tf.name_scope(name):
        BATCH_SIZE, NUM_FRAMES, HEIGHT, WIDTH, NUM_CHANNELS = x_input.get_shape().as_list()
        NUM_CLASSES = 500

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


def backend_resnet(x_input, resnet_size=34, num_classes=512, training=True, name="resnet"):
    with tf.name_scope(name):
        BATCH_SIZE, NUM_FRAMES, HEIGHT, WIDTH, NUM_CHANNELS = x_input.get_shape().as_list()

        # RESNET
        video_input = tf.reshape(x_input, (-1, HEIGHT, WIDTH, NUM_CHANNELS))  # BATCH_SIZE*NUM_FRAMES

        #  = tf.cast(video_input, tf.float32)
        resnet = ResNet(resnet_size=resnet_size, bottleneck=False, num_classes=num_classes, num_filters=64,
                        kernel_size=7, conv_stride=2, first_pool_size=3, first_pool_stride=2,
                        second_pool_size=7, second_pool_stride=1, block_sizes=get_block_sizes(resnet_size),
                        block_strides=[1, 2, 2, 2], final_size=512)
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


def model_step1(x_input):
    """
    Includes:
        - spatiotemporal CNN front end
        - 34 layer resnet
        - fully connected layer (without softmax)
    :param x_input: input tensor
    :return:
    """
    x = frontend_3D(x_input, training=False)
    x = backend_resnet34(x)
    x = concat_resnet_output(x)
    x = fully_connected_logits(x, 500)
    return x


