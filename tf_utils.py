import tensorflow as tf
import os


def set_gpu(gpu_id=0):
    if type(gpu_id) in [list, tuple]:
        gpu_id = [str(i) for i in gpu_id]
    else:
        gpu_id = str(gpu_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id


def start_interactive_session():
    sess = tf.InteractiveSession()
    # with tf.Session() as sess:
    # sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())  # what are local vars?
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    return sess