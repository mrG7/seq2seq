import numpy as np
import pandas as pd
import logging
import os
from multiprocessing import Pool

def find_paths(words):
    paths = []
    for word in words:
        for dataset in datasets:
            try:
                files = [file for file in os.listdir(data_root_dir + "/" + word + "/" + dataset)
                         if file[-len(extension):] == extension]#""npy"]
                for file in files:
                    paths.append([dataset, word, file, data_root_dir + "/" + word + "/" + dataset + "/" + file])
            except:
                logging.debug("%s-%s not saved" % (dataset, word))
    paths = pd.DataFrame(data=paths, columns=["dataset", "word", "filename", "path"])
    return paths


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


if __name__ == "__main__":

    #---------------------------------------------------------------------------------------------#
    # USER INPUT
    data_root_dir = ("/vol/paramonos/projects/mat10/datasets/LRW/Data")
        # ("/vol/atlas/homes/thanos/bbc/lipread_mp4")
        # ("/home/mat10/Documents/MSc Machine Learning/"
        #              "ISO_Deep_Lipreading/Stafylakis_Tzimiropoulos/"
        #              "Tensorflow2/data")

    words = os.listdir(data_root_dir)
    datasets = ['train', 'val', 'test']
    savedir = "/vol/paramonos/projects/mat10/datasets/LRW/Paths"
    LOG_FILENAME = savedir + "/data_transformation_errors.log"
    
    extension = "tfrecords"

    num_cores = 8
    #---------------------------------------------------------------------------------------------#

    logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)

    words_split = split_list(words, num_cores)

    logging.info('Start gathering paths')

    pool = Pool(num_cores)
    paths = pool.map(find_paths, words_split)

    logging.info('Gathering paths completed')

    paths = pd.concat(paths, axis=0)

    unwords = pd.DataFrame(data=paths['word'].unique(), columns=['word'])
    unwords['class'] = np.arange(unwords.shape[0])

    paths = paths.merge(unwords, on=['word'], how='left')
    train_paths = paths[paths['dataset'] == 'train']
    val_paths = paths[paths['dataset'] == 'val']
    test_paths = paths[paths['dataset'] == 'test']

    logging.info('Saving results')
    paths.to_csv(savedir + "/" + "data_info_"  + extension + ".csv", index=False)
    train_paths.to_csv(savedir + "/" + "train_data_info_"  + extension + ".csv", index=False)
    val_paths.to_csv(savedir + "/" +  "val_data_info_"  + extension + ".csv", index=False)
    test_paths.to_csv(savedir + "/" +  "test_data_info_"  + extension + ".csv", index=False)

    logging.info('Saving results finished')

