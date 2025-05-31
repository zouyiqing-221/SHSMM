import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import argparse
from torch import optim
import random

from meta import Meta
from dataset import *
from models import *
from datetime import datetime

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def main(args):
    torch.manual_seed(221)
    torch.cuda.manual_seed_all(221)
    np.random.seed(221)
    random.seed(221)

    args.info_param_map = {'edge_emb_w': 'local_update',
                           'fc_in_w': 'local_update',
                           'fc_in_b': 'local_update',
                           'self_attn_in_proj_w': 'local_update',
                           'self_attn_in_proj_b': 'local_update',
                           'self_attn_out_proj_w': 'local_update',
                           'self_attn_out_proj_b': 'local_update',
                           'linear1_w': 'local_update',
                           'linear1_b': 'local_update',
                           'linear2_w': 'local_update',
                           'linear2_b': 'local_update',
                           'fc_out_w': 'local_update',
                           'fc_combine_w': 'local_update'}

    args.spatial_hierarchy_info_list = args.spatial_hierarchy_info_list.split(', ')
    args.cluster_tree_hierarchy_structure = [int(h) for h in args.cluster_tree_hierarchy_structure.split(', ')]
    args.loss_fn = nn.NLLLoss()

    print(args)

    args.net = SHSMM(lid_in_dim=80, lid_emb_dim=3, gid_in_dim=8000, gid_emb_dim=16,
                     transformer_in_dim=64, transformer_hidden_dim=64, num_heads=1,
                     fc_combine_dim=16,
                     level_num=args.level_num, rid_emb_dim=3, task_rep_dim=64,
                     use_gk=args.use_gk,
                     use_spatial_hierarchy=args.use_spatial_hierarchy,
                     spatial_hierarchy_info_list=args.spatial_hierarchy_info_list,
                     input_cat_destination=args.input_cat_destination, granularity=args.granularity)

    if 'semantic' in args.spatial_hierarchy_info_list:
        args.lrs = Learning_Rate_Scaling(single_level_feature_dim=1000,
                                         per_hierarchy_dim=100,
                                         hidden_dim=64,
                                         cluster_tree_hierarchy_structure=args.cluster_tree_hierarchy_structure,
                                         enhanced=True,
                                         sigma=args.sigma,
                                         activation=args.activation).to(args.device)
        optimizer = optim.Adam([{'params': args.net.parameters(), 'lr': args.beta},
                                {'params': args.lrs.parameters(), 'lr': args.beta}],
                               betas=(args.b1, args.b2))
    else:
        optimizer = optim.Adam([{'params': args.net.parameters(), 'lr': args.beta}], betas=(args.b1, args.b2))

    model = Meta(args).to(args.device)

    tmp = filter(lambda x: x.requires_grad, model.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print('Total trainable tensors:', int(num))

    save_path = f'data/{args.city}/{args.granularity}_split/results'

    train_data_loader = get_data_loader(args.city, args.granularity, batch_size=args.batch_task_num, mode='train', region=args.region)
    val_data_loader = get_data_loader(args.city, args.granularity, batch_size=1, mode='val', region=args.region)
    test_data_loader = get_data_loader(args.city, args.granularity, batch_size=1, mode='test', region=args.region)

    loss_not_decreasing = 0
    best_epoch, best_acc, best_loss = 0, 0, 999
    continue_train_epoch_num = 5
    gamma_dict = torch.ones(args.granularity, args.granularity, 1).to(device)

    for epoch in range(args.max_epoch):

        # Training
        model.train()
        loss = 0
        min_dist_corrects = [0 for _ in range(args.K + 1)]
        corrects = [0 for _ in range(args.K + 1)]
        query_sz = 0
        step_time = datetime.now()
        total_loss_gap = 0

        for step, train_task_batch in enumerate(train_data_loader):
            batch_min_dist_corrects, batch_corrects, batch_loss, batch_query_sz, gamma_dict = model(train_task_batch, gamma_dict, training=True,
                                                                                                    epoch=epoch)

            optimizer.zero_grad()
            batch_loss.backward(retain_graph=True)
            optimizer.step()

            for j in range(args.K + 1):
                min_dist_corrects[j] += batch_min_dist_corrects[j]
                corrects[j] += batch_corrects[j]
            query_sz += batch_query_sz
            train_task_num = len(train_task_batch)
            loss = loss + batch_loss.item() * train_task_num

        print(f'\nTraining Query Size: {query_sz}')
        min_dist_accs = np.array(min_dist_corrects) / query_sz
        accs = np.array(corrects) / query_sz
        loss = loss / query_sz
        print("\n=================meta-training time cost: {}=================".format(datetime.now() - step_time))
        print(f'\nEPOCH {epoch} \nMIN-ACCS      ', end='[ ')
        for k in min_dist_accs:
            print(f'{k:.2%}', end=' ')
        print('] \nSMM-ACCS      ', end='[ ')
        for k in accs:
            print(f'{k:.2%}', end=' ')
        print(f'] \tSMM_LOSS      {loss:.4f}')

        # Validation
        model.eval()
        loss_val = 0
        min_dist_corrects = [0 for _ in range(args.K + 1)]
        corrects = [0 for _ in range(args.K + 1)]
        query_sz = 0
        step_time = datetime.now()

        for val_task_batch in val_data_loader:

            batch_min_dist_corrects, batch_corrects, batch_loss, batch_query_sz, _ = model(val_task_batch, training=False)

            for j in range(args.K + 1):
                min_dist_corrects[j] += batch_min_dist_corrects[j]
                corrects[j] += batch_corrects[j]
            query_sz += batch_query_sz
            loss_val = loss_val + batch_loss.item()

        print(f'\nValidating Query Size: {query_sz}')
        min_dist_accs = np.array(min_dist_corrects) / query_sz
        accs = np.array(corrects) / query_sz
        loss_val = loss_val / query_sz
        print("\n=================meta-validating time cost: {}=================".format(datetime.now() - step_time))
        print(f'\nEPOCH {epoch} \nMIN-ACCS      ', end='[ ')
        for k in min_dist_accs:
            print(f'{k:.2%}', end=' ')
        print('] \nSMM-ACCS      ', end='[ ')
        for k in accs:
            print(f'{k:.2%}', end=' ')
        print(f'] \tSMM_LOSS      {loss_val:.4f}')

        if loss_val < best_loss:
            best_epoch = epoch
            best_acc = accs[-1]
            best_loss = loss_val
            loss_not_decreasing = 0
            torch.save(model.state_dict(), f'{save_path}/SH-SMM.params')
            save_dict(gamma_dict.to(torch.device('cpu')), f'data/{args.city}/{args.granularity}_split/gamma_dict')
        else:
            if epoch >= 10:
                loss_not_decreasing += 1

        print(f'\n=========BEST EPOCH\t{best_epoch}\tMAX ACC\t{best_acc:.4%}\tMIN LOSS\t{best_loss:.8f}=========\n')

        # Testing
        if loss_not_decreasing >= continue_train_epoch_num:

            loss_test = 0
            min_dist_corrects = [0 for _ in range(args.K + 1)]
            corrects = [0 for _ in range(args.K + 1)]
            query_sz = 0
            step_time = datetime.now()

            model = Meta(args).to(args.device)
            model.load_state_dict(torch.load(f'{save_path}/SH-SMM.params'))
            # update_lr_schedule_dict = pd.read_pickle(f'{prefix}/{data_file}/meta/{args.granularity}_split/update_lr_schedule_dict.pkl')
            model.eval()

            for test_task_batch in test_data_loader:

                batch_min_dist_corrects, batch_corrects, batch_loss, batch_query_sz, _ = model(test_task_batch, training=False)

                for j in range(args.K + 1):
                    min_dist_corrects[j] += batch_min_dist_corrects[j]
                    corrects[j] += batch_corrects[j]
                query_sz += batch_query_sz
                loss_test = loss_test + batch_loss.item()

            print(f'\nTesting Query Size: {query_sz}')
            min_dist_accs = np.array(min_dist_corrects) / query_sz
            accs = np.array(corrects) / query_sz
            loss_test = loss_test / query_sz
            print("\n=================meta-testing time cost: {}=================".format(datetime.now() - step_time))
            print(f'\nEPOCH {epoch} \nMIN-ACCS      ', end='[ ')
            for k in min_dist_accs:
                print(f'{k:.2%}', end=' ')
            print('] \nSMM-ACCS      ', end='[ ')
            for k in accs:
                print(f'{k:.2%}', end=' ')
            print(f'] \tSMM_LOSS      {loss_test:.4f}')

            break

    print('\n---FINISH---')


def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=device)
    # parser.add_argument('--prefix', type=str, default=prefix)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--granularity', type=int, choices={128, 256, 512, 1024, 2048}, default=512)
    parser.add_argument('--is_meta', type=bool, default=True)
    parser.add_argument('--batch_task_num', type=int, default=32)
    parser.add_argument('--alpha', type=float, default=1e-2, help='local update learning rate')
    parser.add_argument('--beta', type=float, default=1e-3, help='global update learning rate')
    parser.add_argument('--K', type=int, default=5, help='local update step number')
    parser.add_argument('--b1', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of second order momentum of gradient')
    parser.add_argument('--use_gk', type=bool, default=True)
    parser.add_argument('--use_spatial_hierarchy', type=bool, default=True)
    parser.add_argument('--spatial_hierarchy_info_list', type=str, default='geographical, semantic',
                        choices={'geographical, semantic',
                                 'geographical',
                                 'semantic'})
    parser.add_argument('--level_num', type=int, choices={1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, default=4)
    parser.add_argument('--feature_level_num', type=int, choices={1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, default=10)
    parser.add_argument('--cluster_tree_hierarchy_structure', type=str, default='4, 4, 4, 1')
    parser.add_argument('--osm_tree_hidden_dim', type=int, default=64)
    parser.add_argument('--city', type=str, default='xian', choices={'xian', 'chengdu'})
    parser.add_argument('--region', type=str, default='XIAN', choices={'ZhongLou', 'DaYanTa', 'Middle', 'XIAN', 'Test'},
                        help="To use the whole region, choose 'XIAN', others are sub-regions of XIAN.")
    parser.add_argument('--eta', type=float, default=0.03,
                        help='soft dropout rate to dropout alpha; '
                             'for a sub-region, eg. ZhongLou, better set eta to be a larger number, eg. 0.05/0.07/0.10/...')
    parser.add_argument('--sigma', type=float, default=20.0)
    parser.add_argument('--lambda_', type=float, default=1/10, help='parameter of the exponential distribution')
    parser.add_argument('--activation', type=str, choices={'sech', 'sigmoid'}, default='sech',
                        help='sigmoid may make LRS useless, you can test out.')
    parser.add_argument('--input_cat_destination', type=bool, default=False,
                        help='For SMM, set False; for SMMD, set True.')
    args = parser.parse_args()

    main(args)