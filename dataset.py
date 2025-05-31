import torch
import os
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class SptQryDataSet(Dataset):
    def __init__(self, x_spts, y_spts, x_qrys, y_qrys, osm_feature_dict, tid_dict):
        super(SptQryDataSet, self).__init__()

        self.x_spts = x_spts
        self.y_spts = y_spts
        self.x_qrys = x_qrys
        self.y_qrys = y_qrys
        self.osm_feature_dict = osm_feature_dict
        self.tid_dict = tid_dict

    def __getitem__(self, index):
        tid = self.tid_dict[index]
        return {'tid': tid,
                'x_spt': self.x_spts[tid],
                'y_spt': self.y_spts[tid],
                'x_qry': self.x_qrys[tid],
                'y_qry': self.y_qrys[tid],
                'semantic_dict': self.osm_feature_dict[tid]}

    def __len__(self):
        return len(self.tid_dict)


class SptQryDataLoader:
    def __init__(self, dataset, batch_size: int = 32, shuffle=True):
        super(SptQryDataLoader, self).__init__()

        self.dataset = dataset
        self.batch_task_num = batch_size
        self.total_task_num = len(dataset)
        self.cursor = -1
        self.id_list = list(range(self.total_task_num))
        self.shuffle = shuffle

    def __len__(self):
        return self.total_task_num

    def __next__(self):
        if self.cursor == -1:
            if self.shuffle:
                np.random.shuffle(self.id_list)
        if self.cursor == -100:
            self.cursor = -1
            raise StopIteration
        start_id = self.cursor + 1
        end_id = self.cursor + 1 + self.batch_task_num
        if end_id < self.total_task_num:
            task_batch = [self.dataset[i] for i in self.id_list[start_id:end_id]]
            self.cursor = end_id - 1
        else:
            task_batch = [self.dataset[i] for i in self.id_list[start_id:self.total_task_num]]
            self.cursor = -100
        return task_batch

    def __iter__(self):
        return self


def hierarchy_region_id(task_id, raw_granularity, target_granularity_list):
    target_tid_list = []
    target_row_id_list = []
    target_col_id_list = []
    for g in target_granularity_list:
        row_id = ((task_id - 1) // raw_granularity) // (raw_granularity / g)
        col_id = ((task_id - 1) % raw_granularity) // (raw_granularity / g)
        tid_g = row_id * g + col_id + 1
        target_tid_list.append(int(tid_g - 1))
        target_row_id_list.append(int(row_id))
        target_col_id_list.append(int(col_id))

    return target_tid_list, target_row_id_list, target_col_id_list


def semantic_knowledge_preprocess(tid, granularity, task_osm_feature, osm_task_avg_dict, device, feature_level_num=10):

    row_id = ((tid - 1) // granularity)
    col_id = ((tid - 1) % granularity)
    task_rep = task_osm_feature[row_id][col_id].to(device)

    granularity_list = [1, 1, 1, 1, 1]
    coarse_granularity = 1
    while coarse_granularity < granularity:
        granularity_list.append(coarse_granularity)
        coarse_granularity *= 2
    granularity_list = granularity_list[- (feature_level_num - 1):]

    region_avg_list = []
    target_tid_list, target_row_id_list, target_col_id_list = hierarchy_region_id(tid, granularity, granularity_list)

    for row_i, col_j, k in zip(target_row_id_list, target_col_id_list, granularity_list):
        region_avg = osm_task_avg_dict[k][row_i][col_j].to(device)
        region_avg_list.append(region_avg)

    region_avg_list.append(task_rep)
    osm_region_avg = torch.stack(region_avg_list).unsqueeze(dim=0).to(device)

    semantic_knowledge = {}
    semantic_knowledge['task_level_semantics'] = task_rep
    semantic_knowledge['multi_level_semantics'] = osm_region_avg

    return semantic_knowledge


def task_data_set(data_set_path, edge_dict, device, is_null):

    x, y = {}, {}

    if is_null:
        x['x_dist'] = torch.tensor([]).to(device)
        x['x_local_eid'] = torch.tensor([]).to(device)
        x['x_global_eid'] = torch.tensor([]).to(device)
        x['x_dest_dir'] = torch.tensor([]).to(device)
        x['data_length'] = []
        y['y_label'] = torch.tensor([]).to(device)
    else:
        with open(os.path.join(data_set_path, "X.pkl"), "rb") as f:
            X = pickle.load(f)
        with open(os.path.join(data_set_path, "Y.pkl"), "rb") as f:
            Y = pickle.load(f)
        spt_size = len(Y)

        X_dist_list = [np.vstack([dist for dist in d["dist_list"]]) for d in X]
        X_dist_list = [torch.tensor(dist_list).type(torch.float).to(device) for dist_list in X_dist_list]
        X_edge_global_id_list = [np.vstack([edge for edge in d["edge_idx_list"]]) for d in X]
        X_edge_global_id_list = [torch.tensor(edge_id_list).type(torch.long).to(device) for edge_id_list in X_edge_global_id_list]
        X_edge_id_list = [np.vstack([edge_dict['{}'.format(edge)] for edge in d["edge_idx_list"]]) for d in X]
        X_edge_id_list = [torch.tensor(edge_id_list).type(torch.long).to(device) for edge_id_list in X_edge_id_list]
        X_dest_dir_list = [np.vstack([dest_dir for dest_dir in d["src_dest_direction_list"]]) for d in X]
        X_dest_dir_list = [torch.tensor(dest_dir_list).type(torch.float).to(device) for dest_dir_list in X_dest_dir_list]

        Y = torch.tensor(Y).to(device)

        data_length = [len(d) for d in X_dist_list]
        X_dist_list = pad_sequence(list(X_dist_list), batch_first=True, padding_value=100)
        X_edge_id_list = pad_sequence(list(X_edge_id_list), batch_first=True, padding_value=0)
        X_edge_global_id_list = pad_sequence(list(X_edge_global_id_list), batch_first=True, padding_value=0)
        X_dest_dir_list = pad_sequence(list(X_dest_dir_list), batch_first=True, padding_value=-1)
        Y = torch.stack(list(Y), dim=0)

        x['x_dist'] = X_dist_list
        x['x_local_eid'] = X_edge_id_list
        x['x_global_eid'] = X_edge_global_id_list
        x['x_dest_dir'] = X_dest_dir_list
        x['data_length'] = data_length
        y['y_label'] = Y

    return x, y


def get_data_loader(city, granularity, batch_size=32, mode='train', region=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_path = f'data/{city}/{granularity}_split'
    edge_dicts = pd.read_pickle(f'{data_path}/edge_dictionary.pkl')
    data_info = pd.read_csv(f'{data_path}/data_info.csv', index_col=0)

    x_spts, y_spts, x_qrys, y_qrys, semantic_knowledge_dict, tid_dict = {}, {}, {}, {}, {}, {}

    task_num = 0

    if region == 'XIAN' or region == 'CHENGDU':
        sub_area_for_test = 0
    else:
        sub_area_for_test = 1

    if region == 'ZhongLou':
        x1, x2, y1, y2 = 38, 70, 29, 52     # ZhongLou
    elif region == 'DaYanTa':
        x1, x2, y1, y2 = 62, 120, 55, 120   # DaYanTa
    elif region == 'Middle':
        x1, x2, y1, y2 = 43, 96, 31, 81     # Middle
    elif region == 'Test':
        x1, x2, y1, y2 = 80, 84, 21, 22     # Test

    lat1, lat2, lng1, lng2 = 0, 0, 0, 0
    if sub_area_for_test == 1:
        if granularity == 64:
            lat1, lat2, lng1, lng2 = granularity - 0.5 * x2, granularity - 0.5 * x1, 0.5 * y1, 0.5 * y2
        if granularity == 128:
            lat1, lat2, lng1, lng2 = granularity - 1 * x2, granularity - 1 * x1, 1 * y1, 1 * y2
        if granularity == 256:
            lat1, lat2, lng1, lng2 = granularity - 2 * x2, granularity - 2 * x1, 2 * y1, 2 * y2
        if granularity == 512:
            lat1, lat2, lng1, lng2 = granularity - 4 * x2, granularity - 4 * x1, 4 * y1, 4 * y2
        if granularity == 1024:
            lat1, lat2, lng1, lng2 = granularity - 8 * x2, granularity - 8 * x1, 8 * y1, 8 * y2

    task_osm_feature = pd.read_pickle(f'{data_path}/task_osm_feature.pkl')
    osm_task_avg_dict = pd.read_pickle(f'{data_path}/{mode}_task_avg_dictionary.pkl')

    data_path = f'{data_path}/meta_{mode}_tasks'
    for filename in tqdm(os.listdir(data_path)):

        tid = int(filename.lstrip("task"))

        if sub_area_for_test == 1:
            if tid <= lat1 * granularity or tid > lat2 * granularity:
                continue
            if tid % granularity <= lng1 or tid % granularity > lng2:
                continue

        task_info = data_info.loc[tid]
        if task_info[f'meta_{mode}_query'] == 0:
            continue
        if task_info[f'meta_{mode}_support'] == 0:
            nonzero_spt = False
        else:
            nonzero_spt = True

        edge_dict = edge_dicts[filename]
        support_set_path = f'{data_path}/{filename}/support'
        query_set_path = f'{data_path}/{filename}/query'

        # Support Data Set
        if nonzero_spt:
            x_spt, y_spt = task_data_set(support_set_path, edge_dict, device, is_null=False)
        else:
            x_spt, y_spt = task_data_set(support_set_path, edge_dict, device, is_null=True)
        x_spt['nonzero_spt'] = nonzero_spt

        x_qry, y_qry = task_data_set(query_set_path, edge_dict, device, is_null=False)

        semantic_knowledge = semantic_knowledge_preprocess(tid, granularity, task_osm_feature, osm_task_avg_dict, device)

        x_spts[tid] = x_spt
        y_spts[tid] = y_spt
        x_qrys[tid] = x_qry
        y_qrys[tid] = y_qry
        semantic_knowledge_dict[tid] = semantic_knowledge
        tid_dict[task_num] = tid

        task_num += 1

    dataset = SptQryDataSet(x_spts, y_spts, x_qrys, y_qrys, semantic_knowledge_dict, tid_dict)

    if mode == 'train':
        is_shuffle = True
    else:
        is_shuffle = False

    data_loader = SptQryDataLoader(dataset=dataset, batch_size=batch_size, shuffle=is_shuffle)

    print(f'{task_num}-tasks-in-total')

    return data_loader