import argparse
import copy
import math
import os
import random
import pandas as pd
import numpy as np
import pickle

import torch

from tqdm import tqdm

from utils import *


def get_train_val_test_sets(Z, train_rate, val_rate, seed):
    num_sets = len(Z)
    idxes = np.random.RandomState(seed=seed).permutation(num_sets)
    train_size = round(num_sets * train_rate)
    val_size = round(num_sets * val_rate)
    test_size = num_sets - train_size - val_size
    support_size = round(train_size * (train_rate / (train_rate + val_rate)))

    train_spt_sets = [Z[spt_idx] for spt_idx in idxes[:support_size]]
    train_qry_sets = [Z[qry_idx] for qry_idx in idxes[support_size:train_size]]
    train_sets = [Z[train_idx] for train_idx in idxes[:train_size]]
    val_sets = [Z[val_idx] for val_idx in
                    idxes[train_size: train_size + val_size]]
    test_sets = [Z[test_idx] for test_idx in idxes[train_size + val_size:]]
    return train_spt_sets, train_qry_sets, \
           train_sets, val_sets, test_sets


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='xian', choices={'xian', 'chengdu'})
    parser.add_argument('--granularity', type=int, default=512, choices={128, 256, 512, 1024})
    opt = parser.parse_args()

    train_rate = 0.8
    val_rate = 0.1

    city = opt.city
    granularity = opt.granularity

    seed = 2024

    if city == 'xian':
        mbr = MBR(34.20, 108.91, 34.29, 109.01)
    elif city == 'chengdu':
        mbr = MBR(30.647, 104.033, 30.737, 104.133)
    height = mbr.get_h()
    width = mbr.get_w()
    S = height * width
    print("Height: {}m".format(height))
    print("Width: {}m".format(width))
    print("S: {}m^2".format(S))

    data_result_save_path = f'data/{city}/'
    meta_learning_sample_path = f'data/{city}/'
    cleaned_data_path = f'data/{city}/cleaned_data'
    os.makedirs(meta_learning_sample_path, exist_ok=True)

    X_div = [[]] * (granularity ** 2)
    Y_div = [[]] * (granularity ** 2)
    Z_div = [[]] * (granularity ** 2)
    X = []
    Y = []
    Z = []

    time_format = '%Y,%m,%d,%H,%M,%S'
    taxi_id_list = []
    edge_id_list = []
    dest_edge_id_list = []
    dest_edge_id = 0
    grid_sample_num = [0] * (granularity ** 2)
    grid_dict = {}
    for i in list(range((granularity ** 2))):
        grid_dict["{}".format(i + 1)] = 0

    X_div = pd.read_pickle(os.path.join(cleaned_data_path, 'X.pkl'))
    Y_div = pd.read_pickle(os.path.join(cleaned_data_path, 'Y.pkl'))
    [[x.update(y) for (x, y) in zip(X_div[grid_id], Y_div[grid_id])] for grid_id in range(512 ** 2)]

    sample_dense_grid_set = []
    for grid_id in tqdm(range(granularity ** 2)):
        sample_dense_grid_set.extend(X_div[grid_id])

    print('---Finish Cleaning!---\n')

    train_spt_sets, train_qry_sets, train_sets, val_sets, test_sets = get_train_val_test_sets(sample_dense_grid_set, train_rate, val_rate, seed)


    split_num_list = [granularity]
    for split_num in split_num_list:
        data = {'grid_ID': [], 'total': [],
                'meta_train_support': [], 'meta_train_query': [],
                'meta_val_support': [], 'meta_val_query': [],
                'meta_test_support': [], 'meta_test_query': []}
        grid = Grid(mbr, split_num, split_num)

        train_spt_X = [[]] * (split_num ** 2)
        train_spt_Y = [[]] * (split_num ** 2)
        for train_spt_set in train_spt_sets:
            lat = train_spt_set['lat']
            lng = train_spt_set['lng']
            row_idx, col_idx = grid.get_idx(lat, lng)
            grid_id = row_idx * split_num + col_idx
            if len(train_spt_X[grid_id]) == 0:
                train_spt_X[grid_id] = [train_spt_set]
                train_spt_Y[grid_id] = [train_spt_set['id']]
            else:
                train_spt_X[grid_id].append(train_spt_set)
                train_spt_Y[grid_id].append(train_spt_set['id'])

        train_qry_X = [[]] * (split_num ** 2)
        train_qry_Y = [[]] * (split_num ** 2)
        for train_qry_set in train_qry_sets:
            lat = train_qry_set['lat']
            lng = train_qry_set['lng']
            row_idx, col_idx = grid.get_idx(lat, lng)
            grid_id = row_idx * split_num + col_idx
            if len(train_qry_X[grid_id]) == 0:
                train_qry_X[grid_id] = [train_qry_set]
                train_qry_Y[grid_id] = [train_qry_set['id']]
            else:
                train_qry_X[grid_id].append(train_qry_set)
                train_qry_Y[grid_id].append(train_qry_set['id'])

        train_X = [[]] * (split_num ** 2)
        train_Y = [[]] * (split_num ** 2)
        for train_set in train_sets:
            lat = train_set['lat']
            lng = train_set['lng']
            row_idx, col_idx = grid.get_idx(lat, lng)
            grid_id = row_idx * split_num + col_idx
            if len(train_X[grid_id]) == 0:
                train_X[grid_id] = [train_set]
                train_Y[grid_id] = [train_set['id']]
            else:
                train_X[grid_id].append(train_set)
                train_Y[grid_id].append(train_set['id'])

        val_X = [[]] * (split_num ** 2)
        val_Y = [[]] * (split_num ** 2)
        for val_set in val_sets:
            lat = val_set['lat']
            lng = val_set['lng']
            row_idx, col_idx = grid.get_idx(lat, lng)
            grid_id = row_idx * split_num + col_idx
            if len(val_X[grid_id]) == 0:
                val_X[grid_id] = [val_set]
                val_Y[grid_id] = [val_set['id']]
            else:
                val_X[grid_id].append(val_set)
                val_Y[grid_id].append(val_set['id'])

        test_X = [[]] * (split_num ** 2)
        test_Y = [[]] * (split_num ** 2)
        for test_set in test_sets:
            lat = test_set['lat']
            lng = test_set['lng']
            row_idx, col_idx = grid.get_idx(lat, lng)
            grid_id = row_idx * split_num + col_idx
            if len(test_X[grid_id]) == 0:
                test_X[grid_id] = [test_set]
                test_Y[grid_id] = [test_set['id']]
            else:
                test_X[grid_id].append(test_set)
                test_Y[grid_id].append(test_set['id'])

        os.makedirs(os.path.join(meta_learning_sample_path, "{}_split".format(split_num)), exist_ok=True)
        for i in range(split_num ** 2):
            data['grid_ID'].append(i + 1)
            data['total'].append(len(train_X[i]) + len(val_X[i]) + len(test_X[i]))
            data['meta_train_support'].append(len(train_spt_X[i]))
            data['meta_train_query'].append(len(train_qry_X[i]))
            data['meta_val_support'].append(len(train_X[i]))
            data['meta_val_query'].append(len(val_X[i]))
            data['meta_test_support'].append(len(train_X[i]))
            data['meta_test_query'].append(len(test_X[i]))

            if len(train_qry_X[i]) > 0:
                meta_training_save_path = f'{meta_learning_sample_path}/{granularity}_split/meta_train_tasks/task{i+1}'
                os.makedirs(meta_training_save_path, exist_ok=True)
                train_support_path = os.path.join(meta_training_save_path, 'support')
                os.makedirs(train_support_path, exist_ok=True)
                with open(os.path.join(train_support_path, 'X.pkl'), 'wb') as f:
                    pickle.dump(train_spt_X[i], f)
                with open(os.path.join(train_support_path, 'Y.pkl'), 'wb') as f:
                    pickle.dump(train_spt_Y[i], f)

                train_query_path = os.path.join(meta_training_save_path, 'query')
                os.makedirs(train_query_path, exist_ok=True)
                with open(os.path.join(train_query_path, 'X.pkl'), 'wb') as f:
                    pickle.dump(train_qry_X[i], f)
                with open(os.path.join(train_query_path, 'Y.pkl'), 'wb') as f:
                    pickle.dump(train_qry_Y[i], f)

            if len(test_X[i]) > 0:
                meta_testing_save_path = f'{meta_learning_sample_path}/{granularity}_split/meta_test_tasks/task{i+1}'
                os.makedirs(meta_testing_save_path, exist_ok=True)

                test_support_path = os.path.join(meta_testing_save_path, 'support')
                os.makedirs(test_support_path, exist_ok=True)
                with open(os.path.join(test_support_path, 'X.pkl'), 'wb') as f:
                    pickle.dump(train_X[i], f)
                with open(os.path.join(test_support_path, 'Y.pkl'), 'wb') as f:
                    pickle.dump(train_Y[i], f)

                test_query_path = os.path.join(meta_testing_save_path, 'query')
                os.makedirs(test_query_path, exist_ok=True)
                with open(os.path.join(test_query_path, 'X.pkl'), 'wb') as f:
                    pickle.dump(test_X[i], f)
                with open(os.path.join(test_query_path, 'Y.pkl'), 'wb') as f:
                    pickle.dump(test_Y[i], f)

            if len(val_X[i]) > 0:
                meta_validating_save_path = f'{meta_learning_sample_path}/{granularity}_split/meta_val_tasks/task{i+1}'
                os.makedirs(meta_validating_save_path, exist_ok=True)

                val_support_path = os.path.join(meta_validating_save_path, 'support')
                os.makedirs(val_support_path, exist_ok=True)
                with open(os.path.join(val_support_path, 'X.pkl'), 'wb') as f:
                    pickle.dump(train_X[i], f)
                with open(os.path.join(val_support_path, 'Y.pkl'), 'wb') as f:
                    pickle.dump(train_Y[i], f)

                val_query_path = os.path.join(meta_validating_save_path, 'query')
                os.makedirs(val_query_path, exist_ok=True)
                with open(os.path.join(val_query_path, 'X.pkl'), 'wb') as f:
                    pickle.dump(val_X[i], f)
                with open(os.path.join(val_query_path, 'Y.pkl'), 'wb') as f:
                    pickle.dump(val_Y[i], f)

        df = pd.DataFrame(data)
        df.index = [i for i in df['grid_ID']]
        df.to_csv(f'data/{city}/{granularity}_split/data_info.csv')

        print(f'---{split_num} SPLIT META DATA PREPROCESSING FINISH---')