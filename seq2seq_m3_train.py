import tensorflow as tf
from models import Model3
from tf_utils import set_gpu, start_interactive_session
# from metrics import char_edit_dist, word_edit_dist
# import os

# set_gpu(0)

options = {

    'mode': "train",  # "train" or "test
    'dataset': "LRS2",  # name of the dataset, either "LRW" or "LRS2"
    'data_root_dir': "/home/mat10/Documents/MSc_Machine_Learning/MSc_Project/lipreading/Data/LRS/lrs_v1",
    #    "/home/mat10/Documents/MSc_Machine_Learning/MSc_Project/Data/LRS/lrs_v0",  # root directory of the
    # data. contains Data, Paths, Logs
    'data_dir': "example",  # data paths file is in ~/Paths, named <data_dir> + "_data_info_tfrecords.csv"

    'batch_size': 1,   # number of examples in queue either for training or inference
    'frame_size': 118,  # spatial resolution of frames saved tfrecords files
    'crop_size': 112,  # spatial resolution of inputs to model
    'random_crop': True,  # boolean. if True a random elif False a center crop_size window is cropped
    'num_channels': 1,  # number of channels of tfrecords files
    'time_window_len': 1,  # number of consecutive frames concatenated as channels passed as encoder_inputs

    'num_classes': 29,  # number of output classes 29 = |a-z, " ", <sos>, <eos>|
    'max_out_len_multiplier': 1.0,  # max_out_len = max_out_len_multiplier * max_in_len

    'horizontal_flip': True,  # data augmentation. flip horizontally with p=0.5
    'shuffle': True,  # shuffle data paths before queue
    'reverse_time': True,  # return input data in reverse time format
    # (useful when unidir encoder or no attention mechanism is used)

    'frontend_3d': True,
    'resnet_size': 18,  # number of layers in resnet. one of [18, 34, 50 .. ]
    'resnet_num_features': 512,  # size of resnet features. if not 512 a dense layer is added to
    # match desired feature size
    'res_features_keep_prob': 1.0,  # prob of keeping (not dropping) resnet features before encoder

    'encoder_num_layers': 3,  # number of hidden layers in encoder lstm
    'encoder_num_hidden': 512,  # number of hidden units in encoder lstm
    'encoder_dropout_keep_prob' : 1.0,  # probability of keeping neuron
    'residual_encoder': True,
    'bidir_encoder': False,

    'encoder_state_as_decoder_init': False,  # bool. if True encoder state is used for decoder init state, otherwise
    # zero state is used
    'decoder_num_layers': 3,  # number of hidden layers in decoder lstm
    'decoder_num_hidden': 512,  # number of hidden units in decoder lstm
    'decoder_output_num_units': 128,  # number of units in MLP hidden layer from attention cell
    'residual_decoder': False,
    'attention_num_units': 100,
    'attention_layer_size': 600,  # number of hidden units in attention layer,
    # if None, cell output and context vector are concatenated
    'norm_attention_layer': True,
    'reset_global_step': False,
    # 'num_hidden_out': 128,  # number of hidden units in output fcn

    'beam_width': 20,  # number of best solutions used in beam decoder
    'max_in_len': None,  # maximum number of frames in input videos
    'max_out_len': None,  # maximum number of characters in output text

    'num_epochs': 10,  # number of epochs over dataset for training
    'start_epoch': 1,  # epoch to start
    'train_era_step': 1,  # start train step during current era
    'learn_rate': 0.0005,  # initial learn rate corresponing top global step 0, or max lr for Adam
    'ss_prob': 0.0,  # scheduled sampling probability for training. probability of passing decoder output as next
    # decoder input instead of ground truth
    'num_decay_steps': 0.05,  # None: decay once per epoch, float: multiply that with num steps per epoch to get
    'decay_rate': 0.95,

    'restore': False,  # boolean. restore model from disk
    'restore_model': "/data/mat10/MSc_Project/Models/model_2_0state_train/m2_train_0state_ss005_era4_final",  # path to model to restore


    'save': False,  # boolean. save model to disk during current era
    'save_model': "/data/mat10/MSc_Project/Models/model_2_0state_train/m2_train_0state_ss010_era5",
    # "/home/mat10/Desktop/seq2seq_m2/models/model4",  # name for saved model
    'num_models_saved': 100,  # total number of models saved
    'save_steps': 5000,  # every how many steps to save model

    'save_graph': True,
    'save_dir': "/home/mat10/Documents/MSc_Machine_Learning/MSc_Project/lipreading/Python/graphs/graph5",
    'save_summaries': True
          }


# if __name__ == "__main__":

model = Model3(options)
# gv = tf.global_variables()
# tv = tf.trainable_variables()
# ntv = [v for v in gv if v not in tv]
# ntv2 = [v for v in ntv if "Adam" not in v.name]
# mv = tf.model_variables()
# update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

sess = start_interactive_session()

assert options['mode'] == 'train'
model.train(sess, 100)

model.save_graph(sess)
model.save_summaries(sess)


# model.save_model(sess, options['save_model'])

# Train ------------------------------------------------------------------------------- #
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# with tf.Session() as sess:
# sess.run(tf.global_variables_initializer())
# sess.run(tf.local_variables_initializer())  # what are local vars?
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(coord=coord)


# ei = sess.run(model.encoder_inputs)
# _, loss = sess.run([model.update_step, model.train_loss])
# grad = sess.run(model.gradients)

# # Test -------------------------------------------------------------------------------- #
# assert options['mode'] == 'test'
# res = model.predict(sess, 2)
# cer = char_edit_dist(res)
# wer = word_edit_dist(res)







# # import the inspect_checkpoint library
# from tensorflow.python.tools import inspect_checkpoint as chkp
#
# # print all tensors in checkpoint file
# a = chkp.print_tensors_in_checkpoint_file("/home/mat10/Desktop/seq2seq_m2/models/model3_epoch1_step0", tensor_name='', all_tensors=True)
#
#
#
# from tensorflow.python import pywrap_tensorflow
# reader = pywrap_tensorflow.NewCheckpointReader(
#     "/home/mat10/Desktop/seq2seq_m2/models/model3_epoch1_step0")
# var_to_shape_map = reader.get_variable_to_shape_map()



