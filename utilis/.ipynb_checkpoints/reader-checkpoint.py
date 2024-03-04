import os.path

import numpy as np
import pandas as pd

from utilis.constants import KEYS, SPECIAL_KEYS


def csv_reader(file_path, keys=KEYS, special_keys=SPECIAL_KEYS):
    """
    concrete reader methods for dataset.
    key: defines the column names in the file
    """
    data = pd.read_csv(file_path, usecols=keys)
    origin_length = len(data)
    data = data.dropna(axis=0, subset=keys)
    after_drop_length = len(data)
    print(f'Origin length: {origin_length}; Dropna: {after_drop_length}')
    task_name = keys[-1]
    # data = data[:10]
    if os.path.exists(f'auxiliary/{task_name}_auxiliary.csv') and 'train' in file_path:
        data_auxiliary = pd.read_csv(f'auxiliary/{task_name}_auxiliary.csv', usecols=keys)
        data_auxiliary = data_auxiliary.dropna(axis=0, subset=keys)
        print(f'Found auxiliary data for {task_name} {len(data_auxiliary)}')
        data = pd.concat([data, data_auxiliary], axis=0)
        data = data.sample(frac=1.0).reset_index(drop=True)
    # data = data[:10]
    data_ = []
    for key in keys:
        if key in data.columns:
            if key in special_keys:
                data_.append([text.strip().split('>>')[1:] for text in list(data[key])])
            else:
                data_.append([text.strip() for text in list(data[key])])
        else:
            raise KeyError(f"{key} doesn't exist in source csv")

    return data_


def read_vector(word_vector_source, skip_head=True, vector_dim=300) -> dict:
    """

    :param word_vector_source: path of word2vector file
    :param skip_head: (bool)
    :param vector_dim: dimension of vector
    :return: (dict), key word, value vector
    """
    word_vector = {}
    with open(word_vector_source, 'r', encoding='utf-8') as f:
        if skip_head:
            f.readline()
        line = f.readline()
        assert len(line.split()) == vector_dim + 1
        while line:
            word_vector_list = line.split()
            word, vector = word_vector_list[0], word_vector_list[1:]
            if len(vector) == vector_dim:
                vector = [float(num) for num in vector]
                word_vector[word] = vector
            line = f.readline()
        return word_vector


if __name__ == "__main__":
    path = '../dataset/THUCnews/train.csv'
    csv_reader(path)
