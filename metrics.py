import tensorflow as tf
import numpy as np
from tensor2tensor.layers import common_layers
import distance


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def list_map(f, x, flat=False):
    res = list(map(f, x))
    if flat:
        return flatten_list(res)
    return res


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


def remove_chars(s, chars='<eos>', join_char=""):
    """
    Removes unwanted chars from string s
    Important: works because chars are all small letters while decoding is capital
    if chars contains capital letters it will remove unwanted characters
    """

    def in_eos(s, chars=chars):
        if s in chars:
            return True
        return False
    id_rem = list(map(in_eos, s))
    # join back together as str
    s = [s[i] for i, cond in enumerate(id_rem) if not cond]
    s = join_char.join(s)
    if s == "":
        return s
    if s[0]==" " or s[-1]==" ":
        s = remove_end_spaces(s)
    return s


def char_edit_dist(examples):
    label_, pred_ = examples
    if type(label_) == list:
        return list_map(char_edit_dist, zip(label_, pred_))
    # remove unwanted characters and pads
    label_ = remove_chars(label_)
    ref_len = len(label_)
    pred_ = remove_chars(pred_)
    return distance.levenshtein(label_, pred_) / ref_len


def word_edit_dist(examples):
    label_, pred_ = examples
    if type(label_) == list:
        return list_map(word_edit_dist, zip(label_, pred_))
    label_words = list_map(remove_chars, label_.split(" "))
    label_words = [word for word in label_words if word not in ["", " "]]
    pred_words = list_map(remove_chars, pred_.split(" "))
    pred_words = [word for word in pred_words if word not in ["", " "]]
    ref_len = len(label_words)
    return distance.levenshtein(label_words, pred_words) / ref_len



# def remove_end_spaces(s):
#     """
#     Removes spaces at the beggining and end of a string (not intermediate)
#     :param s: str to remove spaces from
#     :return:
#     """
#     i = 0
#     while s[i] == ' ':
#         i += 1
#     j = -1
#     while s[j] == ' ':
#         j -= 1
#     j += 1
#     if j == 0:
#         return s[i:]
#     else:
#         return s[i:j]
#
#
# def remove_chars(s, chars='<eos> ', join_char=""):
#     """
#     Removes unwanted chars from string s
#     Important: works because chars are all small letters while decoding is capital
#     if chars contains capital letters it will remove unwanted characters
#     """
#     def in_eos(s, chars=chars):
#         if s in chars:
#             return True
#         return False
#     id_rem = list(map(in_eos, s))
#     # join back together as str
#     s = join_char.join([s[i] for i in range(len(s)) if not id_rem[i]])
#     return s
#
#
# def char_edit_dist(examples):
#     label_, pred_ = examples
#     if type(label_) == list:
#         return list_map(char_edit_dist, zip(label_, pred_))
#     # remove unwanted characters and pads
#     label_ = remove_chars(label_)
#     ref_len = len(label_)
#     pred_ = remove_chars(pred_)
#     return distance.levenshtein(label_, pred_) / ref_len
#
#
# def word_edit_dist(examples):
#     label_, pred_ = examples
#     if type(label_) == list:
#         return list_map(word_edit_dist, zip(label_, pred_))
#     label_words = list_map(remove_chars, label_.split(" "))
#     pred_words = list_map(remove_chars, pred_.split(" "))
#     ref_len = len(label_words)
#     return distance.levenshtein(label_words, pred_words) / ref_len


# def word_edit_dist(label_, pred_):
#     def split_(s):
#         return s.split(" ")
#     label_words = list_map(remove_chars,
#                            set(list_map(split_, label_, flat=True)))
#     pred_words = list_map(remove_chars,
#                            set(list_map(split_, pred_, flat=True)))
#     words = list(set(label_words + pred_words))
#     words_dict = {word:i for i, word in enumerate(words)}
#
#     label_encoded = get_word_encoding(label_, words_dict)
#     pred_encoded = get_word_encoding(label_, words_dict)
#
#     Levenshtein.distance(label_encoded[0], pred_encoded[0])
#
#
#     list_map(remove_chars, label_)
#
#     # remove unwanted characters and pads
#     label_ = remove_chars(label_)
#     ref_len = len(label_)
#     pred_ = remove_chars(pred_)
#     return Levenshtein.distance(label_, pred_) / ref_len


def sequence_edit_distance(predictions,
                           labels,
                           weights_fn=common_layers.weights_nonzero, beam_decoder=False):
    """Average edit distance, ignoring padding 0s.
    The score returned is the edit distance divided by the total length of
    reference truth and the weight returned is the total length of the truth.
    Args:
    predictions: Tensor of shape [`batch_size`, `length`, 1, `num_classes`] and
        type tf.float32 representing the logits, 0-padded.
    labels: Tensor of shape [`batch_size`, `length`, 1, 1] and type tf.int32
        representing the labels of same length as logits and 0-padded.
    weights_fn: ignored. The weights returned are the total length of the ground
        truth labels, excluding 0-paddings.
    Returns:
    (edit distance / reference length, reference length)
    Raises:
    ValueError: if weights_fn is not common_layers.weights_nonzero.
    """
    if weights_fn is not common_layers.weights_nonzero:
        raise ValueError("Only weights_nonzero can be used for this metric.")
    # get rid of -1
    predictions = tf.clip_by_value(predictions, 0, 30)
    prediction_eos_idx = tf.where(tf.equal(predictions, 28))
    # predictions[prediction_eos_idx].assign(0)
    with tf.variable_scope("edit_distance", values=[predictions, labels]):
        # Transform logits into sequence classes by taking max at every step.
        if not beam_decoder:
            predictions = tf.to_int32(
                tf.squeeze(tf.argmax(predictions, axis=-1), axis=(2)))
        nonzero_idx = tf.where(tf.not_equal(predictions, 0))
        sparse_outputs = tf.SparseTensor(nonzero_idx,
                                         tf.gather_nd(predictions, nonzero_idx),
                                         tf.shape(predictions, out_type=tf.int64))
        labels = tf.squeeze(labels, axis=(2, 3))
        nonzero_idx = tf.where(tf.not_equal(labels, 0))
        label_sparse_outputs = tf.SparseTensor(nonzero_idx,
                                               tf.gather_nd(labels, nonzero_idx),
                                               tf.shape(labels, out_type=tf.int64))
        distance = tf.reduce_sum(
            tf.edit_distance(sparse_outputs, label_sparse_outputs, normalize=False))
        reference_length = tf.to_float(common_layers.shape_list(nonzero_idx)[0])
    return distance / reference_length  # , reference_length



def char_accuracy(target_labels, predictions, target_labels_lengths):
    """
    Character level accuracy for decoder predictions
    :param target_labels:
    :param predictions:
    :param target_labels_lengths:
    :return: accuracy (ratio of sum(correct_pred)/max_seq_len), accuracy2 (accuracy without pads)
    """
    target_weights = tf.sequence_mask(target_labels_lengths,  # +1 for <eos> token
                                      maxlen=None,  # int(target_labels.shape[1]),
                                      dtype=tf.float32)
    accuracy1 = tf.reduce_mean(tf.cast(tf.equal(target_labels, tf.cast(predictions, tf.int32)), tf.float32))
    accuracy2 = tf.reduce_sum(
        tf.cast(tf.equal(target_labels, tf.cast(predictions, tf.int32)), tf.float32) * target_weights) \
                / tf.cast(tf.reduce_sum(target_labels_lengths), tf.float32)
    return accuracy1, accuracy2

def label_accuracy(target_labels, predictions, target_labels_lengths):
    target_weights = tf.sequence_mask(target_labels_lengths,  # +1 for <eos> token
                                      maxlen=None,  # int(target_labels.shape[1]),
                                      dtype=tf.float32)
    accuracy1  = tf.reduce_mean(tf.cast(
        tf.reduce_all(tf.equal(target_labels, tf.cast(predictions, tf.int32)), axis=1), tf.float32))
    accuracy2 = tf.reduce_sum(tf.cast(
        tf.reduce_all(tf.equal(target_labels, tf.cast(predictions, tf.int32)), axis=1), tf.float32) * target_weights) \
        / tf.cast(tf.reduce_sum(target_labels_lengths), tf.float32)
    return accuracy1, accuracy2

def np_char_accuracy(target_labels, predictions):
    """
    target_labels and predictions are numpy arrays
    """
    correct = (target_labels == predictions).astype(int)
    accuracy1 = np.mean(correct)
    # find location of first <pad>. here <pad> has max_id number so we use argmax to get it
    idx_eos = target_labels.argmax(axis=1)
    for i, val in enumerate(correct):
        val[idx_eos[i]:] = 0
    accuracy2 = np.mean(np.sum(correct, axis=1) / idx_eos)
    return accuracy1, accuracy2




# def np_word_accuracy(target_labels, predictions):
#     """
#     target_labels and predictions are numpy arrays
#     """
#     N, T = target_labels.shape
#     correct = (target_labels == predictions).astype(int)
#     correct_word = np.sum(correct, axis=1) == T
#     accuracy1 = np.mean(correct_word)
#     # find loaction of first <pad>. here <pad> has max_id number so we use argmax to get it
#     idx_eos = target_labels.argmax(axis=1)
#     for i, val in enumerate(correct):
#         val[idx_eos[i]:] = 0
#     correct_word = np.sum(correct, axis=1) == idx_eos
#     accuracy2 = np.mean(correct_word)
#     return accuracy1, accuracy2




