import tensorflow as tf
from models import Model1, Model2
from metrics import char_edit_dist, word_edit_dist
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

def start_interactive_session():
    sess = tf.InteractiveSession()
    # with tf.Session() as sess:
    # sess = tf.Session() 
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())  # what are local vars?
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    return sess

options = {

    'mode': "test",  # "train" or "test

    'data_root_dir': "/vol/paramonos/projects/mat10/datasets/BBC_sentences",
    #    "/home/mat10/Documents/MSc_Machine_Learning/MSc_Project/Data/LRS/lrs_v0",  # root directory of the
    # data. contains Data, Paths, Logs
    'data_dir': "pretrain_sample_3x1_words",  # data paths file is in ~/Paths, named <data_dir> + "_data_info_tfrecords.csv"

    'batch_size': 24,   # number of examples in queue either for training or inference
    'frame_size': 118,  # spatial resolution of frames saved tfrecords files
    'crop_size': 112,  # spatial resolution of inputs to model
    'random_crop': True,  # boolean. if True a random elif False a center crop_size window is cropped
    'num_channels': 1,  # number of channels of tfrecords files
    'time_window_len': 5,  # number of consecutive frames concatenated as channels passed as encoder_inputs

    'num_classes': 29,  # number of output classes 29 = |a-z, " ", <sos>, <eos>|
    'max_out_len_multiplier': 1.0,  # max_out_len = max_out_len_multiplier * max_in_len

    'horizontal_flip': False,  # data augmentation. flip horizontally with p=0.5
    'shuffle': False,  # shuffle data paths before queue
    'reverse_time': False,  # return input data in reverse time format (useful when no attention mechanism is used)

    'resnet_size': 18,  # number of layers in resnet. one of [18, 34, 50 .. ]
    'resnet_num_features': 1024,  # size of resnet features
    'res_features_keep_prob': 1.0,  # prob of keeping (not dropping) resnet features before encoder

    'encoder_num_layers': 3,  # number of hidden layers in encoder lstm
    'encoder_num_hidden': 512,  # number of hidden units in encoder lstm
    'encoder_dropout_keep_prob' : 1.0,  # probability of keeping neuron

    'decoder_num_layers': 3,  # number of hidden layers in decoder lstm
    'decoder_num_hidden': 512,  # number of hidden units in decoder lstm
    'encoder_state_as_decoder_init' : True,  # bool. if True encoder state is used for decoder init state, otherwise
    # zero state is used
    'attention_layer_size': 512,  # number of hidden units in attention layer
    'num_hidden_out': 256,  # number of hidden units in output fcn

    'beam_width': 1,  # number of best solutions used in beam decoder
    'max_in_len': None,  # maximum number of frames in input videos
    'max_out_len': None,  # maximum number of characters in output text

    'num_epochs': 1,  # number of epochs over dataset for training
    'start_epoch': 1,  # epoch to start
    'train_era_step': 1,  # start train step during current era
    'learn_rate': 0.001,  # initial learn rate corresponing top global step 0, or max lr for Adam
    'ss_prob': 0.0,  # scheduled sampling probability for training. probability of passing decoder output as next
    # decoder input instead of ground truth

    'restore': True,  # boolean. restore model from disk
    'restore_model': "/data/mat10/MSc_Project/Models/model_2_cl/m2_pretrain_sample_3x1_ss000_era1_final",  # path to model to restore


    'save': False,  # boolean. save model to disk during current era
    'save_model': "/data/mat10/MSc_Project/Models/model_2_cl/m2_pretrain_sample_3x1_ss000_era2",
    # "/home/mat10/Desktop/seq2seq_m2/models/model4",  # name for saved model
    'num_models_saved': 100,  # total number of models saved
    'save_steps': 5000  # every how many steps to save model


          }


# if __name__ == "__main__":

model = Model2(options)
# gv = tf.global_variables()
# tv = tf.trainable_variables()
# ntv = [v for v in gv if v not in tv]
# ntv2 = [v for v in ntv if "Adam" not in v.name]
# mv = tf.model_variables()
# update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

sess = start_interactive_session()


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
    
# assert options['mode'] == 'train'
# model.train(sess)
# ei = sess.run(model.encoder_inputs)
# _, loss = sess.run([model.update_step, model.train_loss])
# grad = sess.run(model.gradients)

# Test -------------------------------------------------------------------------------- #
assert options['mode'] == 'test'
res = model.predict(sess, 2)
cer = char_edit_dist(res)
wer = word_edit_dist(res)







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



