from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
import tensorflow as tf
# from pathlib import Path
# from inception_processing import distort_color


def image_left_right_flip(image):
    return tf.image.flip_left_right(image)


def video_left_right_flip(video):
    return tf.map_fn(image_left_right_flip, video)


def normalize(videos):
    # return videos * (1. / 255.) - 0.5
    return videos/255.


def slice_video(x, dims, time_window=5):

    batch_size, max_time, h, w, num_channels = dims
    num_slices = max_time - time_window + 1

    def get_time_window(start_t, x=x, time_window=time_window):
        x_ = tf.slice(x, [0, start_t, 0, 0, 0], [batch_size, time_window, h, w, num_channels])
        x_ = tf.transpose(x_, [0, 4, 2, 3, 1])
        return x_

    x_ = tf.map_fn(get_time_window, tf.range(num_slices), dtype=tf.float32)
    x_ = tf.reshape(x_, [batch_size, -1, h, w, num_channels*time_window])
    print(x_)
    return x_


def get_lrw_batch(paths, options):
    """Returns a data split of the RECOLA dataset, which was saved in tfrecords format.
    Args:
        split_name: A train/test/valid split name.
    Returns:
        The raw audio examples and the corresponding arousal/valence
        labels.
    """
    shuffle = options['shuffle']
    batch_size = options['batch_size']
    num_classes = options['num_classes']
    crop_size = options['crop_size']
    horizontal_flip = options['horizontal_flip']

    # root_path = Path(dataset_dir) / split_name
    # paths = [str(x) for x in root_path.glob('*.tfrecords')]

    filename_queue = tf.train.string_input_producer(paths, shuffle=shuffle)

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'video': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    )

    video = tf.cast(tf.decode_raw(features['video'], tf.uint8), tf.float32) #/ 255.
    label = features['label']#tf.decode_raw(features['label'], tf.int64)

    # Number of threads should always be one, in order to load samples
    # sequentially.
    videos, labels = tf.train.batch(
        [video, label], batch_size, num_threads=1, capacity=1000, dynamic_pad=True)

    videos = tf.reshape(videos, (batch_size, 29, 118, 118, 1))
    #labels = tf.reshape(labels, (batch_size,  1))
    labels = tf.contrib.layers.one_hot_encoding(labels, num_classes)

    # if is_training:
        # resized_image = tf.image.resize_images(frame, [crop_size, 110])
        # random cropping
    if crop_size is not None:
        videos = tf.random_crop(videos, [batch_size, 29, crop_size, crop_size, 1])
    # random left right flip
    if horizontal_flip:
        sample = tf.random_uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)
        option = tf.less(sample, 0.5)
        videos = tf.cond(option,
                         lambda: tf.map_fn(video_left_right_flip, videos),
                         lambda: tf.map_fn(tf.identity, videos))
            # lambda: video_left_right_flip(videos),
            # lambda: tf.identity(videos))
    videos = normalize(videos) #tf.cast(videos, tf.float32) * (1. / 255.) - 0.5

    return videos, labels


def get_lrs_batch(paths, options):
    """Returns a data split of the RECOLA dataset, which was saved in tfrecords format.
    Args:
        paths: list with paths to data files
        options: dict with data settings
    Returns:
        The raw audio examples and the corresponding arousal/valence
        labels.
    """
    batch_size = options['batch_size']
    frame_size = options['frame_size']
    num_channels = options['num_channels']
    num_classes = options['num_classes']
    crop_size = options['crop_size']
    # max_in_len = options['max_in_len']
    # max_out_len = options['max_out_len']
    time_window_len = options['time_window_len']

    if options['shuffle']:
        shuffle = options['shuffle']
    else:
        shuffle = False

    if options['horizontal_flip']:
        horizontal_flip = options['horizontal_flip']
    else:
        horizontal_flip = False

    if options['random_crop']:
        random_crop = options['random_crop']
    else:
        random_crop = False
    # root_path = Path(dataset_dir) / split_name
    # paths = [str(x) for x in root_path.glob('*.tfrecords')]

    filename_queue = tf.train.string_input_producer(paths, shuffle=shuffle)

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'video': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string),
            'in_seq_len': tf.FixedLenFeature([], tf.int64),
            'out_seq_len': tf.FixedLenFeature([], tf.int64)
        }
    )

    video = tf.cast(tf.decode_raw(features['video'], tf.uint8), tf.float32)  # / 255.
    label = tf.cast(tf.decode_raw(features['label'], tf.uint8), tf.int32)
    in_seq_len = tf.cast(features['in_seq_len'], tf.int32)
    out_seq_len = tf.cast(features['out_seq_len'], tf.int32)

    # perform bucketing with input_length being the single filter (need to add out_length buckets)
    # Number of threads should always be one, in order to load samples sequentially.
    _seq_lens, [encoder_inputs, target_labels, encoder_inputs_lengths, target_labels_lengths] = \
        tf.contrib.training.bucket_by_sequence_length(in_seq_len,
                                                      [video, label, in_seq_len, out_seq_len], batch_size,
                                                      [20, 30, 50, 60, 88, 120, 160, 200, 250],
                                                      num_threads=1, capacity=500, dynamic_pad=True,
                                                      allow_smaller_final_batch=True)
    # encoder_inputs, target_labels, encoder_inputs_lengths, target_labels_lengths = \
    #     tf.train.batch([video, label, in_seq_len, out_seq_len], batch_size,
    #                    num_threads=1, capacity=500, dynamic_pad=True,
    #                    allow_smaller_final_batch=True)

    encoder_inputs = tf.reshape(encoder_inputs, (batch_size,
                                                 tf.reduce_max(encoder_inputs_lengths),
                                                 frame_size,
                                                 frame_size,
                                                 num_channels))
    target_labels = tf.reshape(target_labels, (batch_size, -1))

    # create decoder_inputs
    # add <sos> token
    # decoder_inputs = tf.identity(target_labels)
    sos_slice = tf.constant(options['num_classes'] - 2, dtype=tf.int32, shape=[options['batch_size'], 1])
    decoder_inputs = tf.concat([sos_slice, target_labels], axis=1)
    decoder_inputs = tf.one_hot(decoder_inputs, num_classes)

    if crop_size is not None and random_crop:
        encoder_inputs = tf.random_crop(encoder_inputs, [batch_size,
                                                         tf.reduce_max(encoder_inputs_lengths),
                                                         crop_size,
                                                         crop_size,
                                                         num_channels])
    elif crop_size:
        start_xy = int((frame_size - crop_size) /  2)
        encoder_inputs = tf.slice(encoder_inputs,
                                  [0, 0, start_xy, start_xy, 0],
                                  [batch_size, tf.reduce_max(encoder_inputs_lengths),
                                   crop_size, crop_size, num_channels])

    encoder_inputs = tf.reshape(encoder_inputs, [batch_size, -1, crop_size, crop_size, 1])
    # random left right flip
    if horizontal_flip:
        sample = tf.random_uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)
        option = tf.less(sample, 0.5)
        encoder_inputs = tf.cond(option,
                                 lambda: tf.map_fn(video_left_right_flip, encoder_inputs),
                                 lambda: tf.map_fn(tf.identity, encoder_inputs))
    encoder_inputs = normalize(encoder_inputs)

    # reverse time dimension in input frames. better performance when a unidirectional encoder is used
    if options['reverse_time']:
        encoder_inputs = tf.reverse(encoder_inputs, axis=[1])

    # slicw video to time_window_len consecutive frames with stride 1
    if time_window_len != 1:
        # pad encoder_inputs s.t. each frame is in the same number of slices
        ei_paddings = [[0, 0], [time_window_len-1, time_window_len-1], [0, 0], [0, 0], [0, 0]]
        padded_encoder_inputs = tf.pad(encoder_inputs, ei_paddings, 'CONSTANT', constant_values=0)
        encoder_inputs = slice_video(padded_encoder_inputs,
                                     dims=[batch_size, tf.reduce_max(encoder_inputs_lengths) + 2*(time_window_len - 1),
                                           crop_size, crop_size, num_channels],
                                     time_window=time_window_len)
        encoder_inputs = tf.reshape(encoder_inputs, [batch_size, -1, crop_size, crop_size, time_window_len])
        encoder_inputs_lengths = encoder_inputs_lengths + time_window_len - 1

    return encoder_inputs, target_labels, decoder_inputs, encoder_inputs_lengths, target_labels_lengths


def get_data_paths(options):
    data_info = pd.read_csv(options['data_root_dir'] + "/Paths/" + options['data_dir'] + "_data_info_tfrecords.csv",
                            dtype={'person_id':str, 'video_id':str})#.sample(frac=1)
    print("Total number of train data: %d" % data_info.shape[0])
    data_info['root_dir'] = options['data_root_dir']
    data_info['path'] = data_info['root_dir'] + "/Data/" + options['data_dir'] + "/" \
                        + data_info['person_id'] + "/" + data_info['video_id'] + ".tfrecords"
    data_paths = list(data_info['path'])
    return data_paths


def get_number_of_steps(data_paths, options):
    number_of_steps_per_epoch = len(data_paths) // options['batch_size'] + 1
    if options['mode'] == "train":
        number_of_steps = options['num_epochs'] * number_of_steps_per_epoch
    else:
        number_of_steps = number_of_steps_per_epoch
    return number_of_steps_per_epoch, number_of_steps


def get_lrw_training_data_batch(data_paths, options):
    with tf.variable_scope('training_data'):
        encoder_inputs, target_labels = get_lrw_batch(data_paths, options)
        print("shape of encoder_inputs is %s" % encoder_inputs.get_shape)
        print("shape of target_labels is %s" % target_labels.get_shape)
        # decoder_inputs_lengths = tf.identity(target_labels_lengths, name="decoder_inputs_lengths")  # + 1
        # max_input_len = tf.reduce_max(encoder_inputs_lengths, name="max_input_len")
        return encoder_inputs, target_labels


def get_lrs_training_data_batch(data_paths, options):
    with tf.variable_scope('training_data'):
        encoder_inputs, target_labels, decoder_inputs, encoder_inputs_lengths, target_labels_lengths = \
            get_lrs_batch(data_paths, options)
        print("shape of encoder_inputs is %s" % encoder_inputs.get_shape)
        print("shape of target_labels is %s" % target_labels.get_shape)
        decoder_inputs_lengths = tf.identity(target_labels_lengths, name="decoder_inputs_lengths")  # + 1
        max_input_len = tf.reduce_max(encoder_inputs_lengths, name="max_input_len")
        return encoder_inputs, target_labels, decoder_inputs, encoder_inputs_lengths, \
            target_labels_lengths, decoder_inputs_lengths, max_input_len


def get_lrs_inference_data_batch(data_paths, options):
    # assert restrictions to options due to inference process
    assert options['horizontal_flip'] == False
    with tf.variable_scope('inference_data'):
        encoder_inputs, target_labels, decoder_inputs, encoder_inputs_lengths, target_labels_lengths = \
            get_lrs_batch(data_paths, options)
        print("shape of encoder_inputs is %s" % encoder_inputs.get_shape)
        print("shape of target_labels is %s" % target_labels.get_shape)
        decoder_inputs_lengths = tf.identity(target_labels_lengths)  # + 1
        max_input_len = tf.reduce_max(encoder_inputs_lengths)
        return encoder_inputs, target_labels, decoder_inputs, encoder_inputs_lengths, \
               target_labels_lengths, decoder_inputs_lengths, max_input_len
