# Script to write videos and target sentences in tfrecords format
#   - videos are encoded in uint8
#   - sentences are encoded according to "get_char_encoding_dict(num_dims)". No <sos>, <eos> tokens are added here, this
#     has to be handled by tf pipeline
# v.1.0 no facial landmarks, mout roi is the same cropped window

# from helper_functions import extract_mouth_roi
from multiprocessing import Pool
from shutil import copyfile
import os
import logging
from scipy.io import loadmat
import numpy as np
import cv2
import tensorflow as tf
import pandas as pd
# import matplotlib.pyplot as plt
import glob
from num2words import num2words


# def pad_eos(x, max_len, axis, eos=28):
#     pad_shape = list(x.shape)
#     pad_shape[0] = max_len - x.shape[0]
#     pad = eos * np.ones(pad_shape, dtype='uint8')
#     x_new = np.concatenate((x, pad), axis=axis)
#     return x_new


def get_char_encoding_dict():
    # symbol: id
    #  ' '  : 0 (<pad> symbol)
    #  A-Z  : 1-26
    # <sos> : 27
    # <eos> : 28
    # removed (0123456789,.!?:)
    all_chars = " ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    all_chars = list(all_chars) + ['<sos>', '<eos>']
    # all_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    # all_chars = list(all_chars) + ['<sos>', '<eos>', ' ']
    char_encoding_dict = {}
    num_dims = 1
    if num_dims == 1:
        for i, char in enumerate(all_chars):
            char_encoding_dict[char] = i
    elif num_dims == 2:
        for i, char in enumerate(all_chars):
            value = np.zeros((1, len(all_chars)+2))
            value[0, i] = 1.  # +1 because all zero vector is reserved for <sos> token
            char_encoding_dict[char] = value

    return char_encoding_dict


def decrypt(chars):
    decr = dict((v, k) for k, v in get_char_encoding_dict().items())
    decoded = np.array([[decr[char] for char in chars_] for chars_ in chars])
    return [''.join(decoded_) for decoded_ in decoded]


def encode_target(s):
    num_dims = 1
    # add <eos> token
    chars = list(s)
    chars = chars + ['<eos>']
    if num_dims == 1:
        encoded = np.array([char_encoding_dict1[char] for char in chars])
    # elif num_dims == 2:
    #     encoded = np.concatenate([char_encoding_dict2[char] for char in chars], axis=0)
    else:
        raise ValueError('improper %d number of dimensions for word encoding, '
                         'should be either 1 or 2' % num_dims)
    encoded = encoded.astype('uint8')
    return encoded


def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert2tfrecord_and_save(video, label, save_file_name):

    """
    convert videos and labels to tf serialized data and save to
    TFRecord binary file
    # Args:
    # videos        List of 29 frames np.ndarray
    # labels        vector of one-hot labels per character
    # out_path      File-path for the TFRecords output file.
    """

    # Open a TFRecordWriter for the output-file.
    with tf.python_io.TFRecordWriter(save_file_name) as writer:

        # Pad with <pad> until max_out_len and Convert the image to raw bytes.
        video_bytes = video.tostring()  # pad_eos(video, max_in_len, 0).tostring()
        label_bytes = label.tostring()  # pad_eos(label1, max_out_len, 0).tostring()
        # label2_bytes = label2.tostring()  # pad_eos(label2, max_out_len, 0).tostring()

        # Create a dict with the data we want to save in the
        # TFRecords file. You can add more relevant data here.
        data = \
            {
                'video': wrap_bytes(video_bytes),
                'label': wrap_bytes(label_bytes),
                'in_seq_len': wrap_int64(video.shape[0]),
                'out_seq_len': wrap_int64(label.shape[0])
            }

        # Wrap the data as TensorFlow Features.
        feature = tf.train.Features(feature=data)

        # Wrap again as a TensorFlow Example.
        example = tf.train.Example(features=feature)

        # Serialize the data.
        serialized = example.SerializeToString()

        # Write the serialized data to the TFRecords file.
        writer.write(serialized)



def split_list(list_, num_chunks):
    chunk_size = len(list_) // num_chunks
    # remainder = len(list_) % num_chunks
    res = []
    for i in range(num_chunks):
        if i == num_chunks-1:
            res.append(list_[i*chunk_size:])
        else:
            res.append(list_[i*chunk_size:(i+1)*chunk_size])
    return res


def check_all_lrw_files(root_dir, extension):
    """
    Checks all data are in root_dir
    Follows the structure of the LRW database
    :return: Bool
    """
    words = os.listdir(root_dir)
    datasets = {'train':1000, 'val':50, 'test':50}
    not_ok = []
    for dataset in datasets: # train, test, val
        for word in words: # ABOUT, ...
            file_dir = root_dir + "/" + word + "/" + dataset
            num_paths = len(glob.glob(file_dir + '/**/*.' + extension, recursive=True))
            if num_paths != datasets[dataset]:
                not_ok.append([word, dataset, num_paths])
    return not_ok

def remove_end_spaces(s):
    """
    Removes spaces at the beggining and end of a string (not intermediate)
    :param s: str to remove spaces from
    :return:
    """
    i = 0
    while s[i] == ' ':
        i += 1
    j = -1
    while s[j] == ' ':
        j -= 1
    j += 1
    if j == 0:
        return s[i:]
    else:
        return s[i:j]

def extract_crop_box_mouth_roi(files_dir, file, crop_dur, video_dur, resolution=118):
    mouth_center_x = 80
    mouth_center_y = 80
    # video
    vidcap = cv2.VideoCapture(files_dir + '/' + file + '.mp4')
    success, image = vidcap.read()
    gray_image = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)]
    count = 1
    while success:
        success, image = vidcap.read()
        # print('Read a new frame: ', success)
        if success:
            gray_image.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            count += 1
    gray_image = np.array(gray_image)

    seq_len = gray_image.shape[0]
    seq_dur = video_dur[1] - video_dur[0]
    frame_dur = seq_dur/seq_len

    start_frame = np.floor((crop_dur[0] - video_dur[0])/frame_dur).astype(int)
    end_frame = start_frame + np.ceil((crop_dur[1] - crop_dur[0])/frame_dur).astype(int)

    # mouth roi
    # dimensions of square mouth region, 112 +- 6 pixels for data augmentation later
    dim = resolution // 2

    mouth_roi = gray_image[start_frame:end_frame, mouth_center_y - dim : mouth_center_y + dim,
                           mouth_center_x - dim : mouth_center_x + dim]
    # either save on disk or return array
    # label = data_info[data_info['word'] == word]['class'].values[0]
    return mouth_roi

def replace_symbol(s, sym, sym2=''):
    s = list(s)
    for i, char in enumerate(s):
        if char in sym:
            s[i] = sym2
    return "".join(s)


# s = 'ABCD 123 ABCDEF 2001 AB'
def nums2text(s):
    # find consecutive numbers
    nums = []
    nums_ids = []
    first = True
    for i, char in enumerate(s):
        if char in list("0123456789"):
            if first:
                nums.append([char])
                nums_ids.append([i])
                first = False
            else:
                nums[-1].append(char)
                nums_ids[-1].append(i)
        else:
            first = True
    added_len = 0
    for ids, num in zip(nums_ids, nums):
        number = "".join(num)
        len1 = len(number)
        if 1500 < int(number) < 2100:  # basic criterion to search for years
            converter = 'year'
        else:
            converter = 'ordinal'
        number = num2words(int(number), to=converter)
        number = replace_symbol(number, ['-'])
        number = number.upper()
        len2 = len(number)
        s = s[:ids[0]+added_len] + number + s[ids[-1]+added_len+1:]
        added_len += (len2 - len1)
    return s

def extract_data(id_list=None):
    """
    :param id_list: numpy array of indices from data_info
    :return:
    """
    paths_done = []

    if id_list is None:
        id_list = data_info  # .index.values

    len_ids = id_list.shape[0]

    for time in range(samples_per_video):

        for id_num, id in enumerate(id_list):

            print("%d epoch, file %d of %d" % (time, id_num, len_ids))

            files_dir = data_root_dir + "/" + id[0]
            assert os.path.isdir(files_dir)

            save_dir = save_root_dir + "/" + dataset + "/" + id[0]
            # check if directory for person already exists, otherwise make
            if not os.path.isdir(save_dir):
                try:
                    os.makedirs(save_dir)
                except:
                    if os.path.isdir(save_dir):
                        pass
                    else:
                        print("something went wrong with making %s" % save_dir)
                        raise SystemExit(0)

            # files = [file[:-4] for file in os.listdir(files_dir) if file[-3:] == 'mp4']

            # for i, file in enumerate(files):
            # print("doing file %d of %d" % (i, len(files)))

            # .txt files
            # read label
            cont = pd.read_csv(files_dir + "/" + id[1] + ".txt", sep=':', header=None)


            frames = cont[0].iloc[3:].values
            # some videos might contain fewer words than required
            if frames.shape[0] < num_words:
                print("video has less than %d frames" % num_words) 
                break

            label_ids, crop_dur, video_dur = extract_words(frames, num_words, start_from)

            label = cont[1].iloc[0]
            label = remove_end_spaces(label)
            label = " ".join(label.split(" ")[label_ids[0]:label_ids[1]])
            # check for numbers
            num_ids = [i in label for i in "0123456789"]
            if any(num_ids):
                # replace with letters
                label = nums2text(label)
            remove = [i for i in "',.!?:" if i in label]
            if any(remove):
                # replace with letters
                for sym in remove:
                    label = replace_symbol(label, sym)
            # prepend and append by tokens
            label = list(label)
            # label = ['<sos>'] + label + ['<eos>']
            try:
                label = encode_target(label)
                # label2 = encode_target(label, 2)
            except:
                print("".join(label))
                raise SystemExit(0)

            # copyfile(files_dir + "/" + id[1] + ".txt", save_dir + "/" + id[1] + ".txt")
            # split mp4 to frames and save to save directory
            save_file_name = save_dir + "/" + id[1] + "_%d_%d" % (label_ids[0], label_ids[1]) + ".tfrecords"

            mouth_roi = extract_crop_box_mouth_roi(files_dir, id[1], crop_dur, video_dur, resolution=118)

            convert2tfrecord_and_save(mouth_roi, label, save_file_name)

            paths_done.append([dataset, id[0], id[1] + "_%d_%d" % (label_ids[0], label_ids[1]), mouth_roi.shape[0], len(label)])

    return paths_done


def extract_words(frames, num_words, start_from):
    start = float(frames[0].split(" ")[1])
    end = float(frames[-1].split(" ")[2])
    # duration = end - start
    if start_from == 'start':
        i_ = 0
    elif start_from == 'random':
        i_ = np.random.randint(frames.shape[0] - num_words + 1)
    # words = frames[i_:i_ + num_words]
    start_crop = float(frames[i_].split(" ")[1])
    end_crop = float(frames[i_+num_words-1].split(" ")[2])
    # duration_crop = end_crop - start_crop
    return [i_, i_+num_words], [start_crop, end_crop], [start, end]


if __name__ == '__main__':

    # USER INPUT ---------------------------------------------------------------------------------- #
    # lms_dir = "/vol/atlas/homes/thanos/bbc/landmarks/2017_9_lip_reading/lip_reading_pts"
    # data_root_dir = "/home/mat10/Documents/MSc_Machine_Learning/MSc_Project/Data/LRS/original/mvlrs_v1/pretrain"  #
    # save_root_dir = "/home/mat10/Documents/MSc_Machine_Learning/MSc_Project/Data/LRS/mvlrs_pretrain_v0/Data" #
    # data_info_dir = "/home/mat10/Documents/MSc_Machine_Learning/MSc_Project/Data/LRS/mvlrs_pretrain_v0/Paths"  # "/home/mat10/Documents/MSc_Machine_Learning/MSc_Project/Data/LRS/lrs_v1/Paths"  # "/data/mat10/LRS2/Paths"  # "/data/mat10/LRS"  # "/data/mat10/datasets/LRW/Paths/data_info_mp4.csv"  #
    # log_dir = "/home/mat10/Documents/MSc_Machine_Learning/MSc_Project/Data/LRS/mvlrs_pretrain_v0/Logs"  # "/home/mat10/Documents/MSc_Machine_Learning/MSc_Project/Data/LRS/lrs_v0/Logs"  # "/data/mat10/LRS2/Logs"  # save_root_dir  #
    data_root_dir = "/vol/paramonos/projects/Stavros/BBC_sentences/mvlrs_v1/pretrain"  # "/data/mat10/LRS/example/main" # "/home/mat10/Documents/MSc_Machine_Learning/MSc_Project/Data/LRS/original/mvlrs_v1/main"  #
    save_root_dir = "/data/mat10/datasets/BBC_sentences/Data"  # "/home/mat10/Documents/MSc_Machine_Learning/MSc_Project/Data/LRS/lrs_v1/Data"  # "/data/mat10/LRS2/Data"  #  "/data/mat10/Datasets/BBC_sentences/Data"  #"/data/mat10/LRS/tfrecords" #
    data_info_dir = "/data/mat10/datasets/BBC_sentences/Paths"  # "/home/mat10/Documents/MSc_Machine_Learning/MSc_Project/Data/LRS/lrs_v1/Paths"  # "/data/mat10/LRS2/Paths"  # "/data/mat10/LRS"  # "/data/mat10/datasets/LRW/Paths/data_info_mp4.csv"  #
    log_dir = "/data/mat10/datasets/BBC_sentences/Logs"  #
    samples_per_video = 1
    num_words = 1 
    start_from = 'random'
    num_cores = 4
    # max_in_len = 200
    # max_out_len = 100  #  max number of characters from first to last char (incl. <pad>, excl. <sos>, <eos>)
    # -------------------------------------------------------------------------------------------- #

    char_encoding_dict1 = get_char_encoding_dict()
    # char_encoding_dict2 = get_char_encoding_dict(2)

    datasets = ['pretrain_random_6x6words_1']  # 'val', 'test', 'train']

    LOG_FILENAME = log_dir + '/errors.log'
    logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
    logging.info('Start')

    for dataset in datasets: # train, test, val

        print(dataset)
        # check if directory for dataset already exists, otherwise make
        if not os.path.isdir(save_root_dir + "/" + dataset):
            os.makedirs(save_root_dir + "/" + dataset)

        data_info = pd.read_csv(data_info_dir + '/BBC_sentences_data_info.csv', dtype='str')
        print("dataset has %d examples" % data_info.shape[0])
        # data_info = data_info.iloc[:200]
        data_split = split_list(data_info.values.astype(list), num_cores)

        # save_lrs_data_from_videos()

        pool = Pool(num_cores)
        paths_done = pool.map(extract_data, data_split)

        paths_done = pd.DataFrame(np.concatenate(paths_done),
                        columns=['dataset', 'person_id', 'video_id', 'input_seq_len', 'out_seq_len'])
        paths_done.to_csv(data_info_dir + "/" + dataset + '_data_info_tfrecords.csv', index=False)


