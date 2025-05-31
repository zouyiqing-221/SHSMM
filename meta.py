import math
import os
import copy
import torch
from torch import nn
from torch import optim
from datetime import datetime
from models import *

class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args):
        super(Meta, self).__init__()

        self.device = args.device

        self.alpha = args.alpha     # local update lr
        self.K = args.K             # local update step num

        self.use_spatial_hierarchy = args.use_spatial_hierarchy
        self.spatial_hierarchy_info_list = args.spatial_hierarchy_info_list
        # self.spatial_info_fusion_mode = args.spatial_info_fusion_mode
        # self.use_hierarchical_lr_generator = args.use_hierarchical_lr_generator

        self.granularity = args.granularity

        self.eta = args.eta     # soft dropout rate

        self.net = args.net
        if self.use_spatial_hierarchy:

            if 'geographical' in self.spatial_hierarchy_info_list:
                self.level_num = args.level_num

            if 'semantic' in self.spatial_hierarchy_info_list:
                self.lrs = args.lrs


        self.loss_fn = args.loss_fn

        self.info_param_map = args.info_param_map
        # if self.spatial_info_fusion_mode == 'info-param-map':
        #     self.info_param_map = args.info_param_map
        # else:
        #     self.info_param_map = None
        # self.bernoulli_distribution = torch.distributions.bernoulli.Bernoulli(0.9)
        # self.exponential = torch.distributions.exponential.Exponential(1/10)
        # self.poisson = torch.distributions.poisson.Poisson(20)

        self.exponential = torch.distributions.exponential.Exponential(args.lambda_)

        # self.spatial_chaos_intensity = args.spatial_chaos_intensity
        # if self.spatial_chaos_intensity:
        #     self.pgd = args.pgd
        #     self.grad_project_step = 3
        #     self.epsilon = 0.1
        #     self.alpha = 0.03

        # self.sdp_trend = args.sdp_trend
        # self.sdp_interval = args.sdp_interval
        #
        # self.use_dropgrad = args.use_dropgrad
        # if self.use_dropgrad:
        #     self.dropgrad_method = args.dropgrad_method
        #     self.drop_grad = DropGrad(method=self.dropgrad_method, rate=0.1, schedule='constant')


    def forward(self, task_batch, gamma_dict=None, training=True, epoch=0):

        task_num = len(task_batch)

        spt_sz = 0
        qry_sz = 0

        losses_qry = torch.tensor([0. for _ in range(self.K + 1)]).to(self.device)  # losses_q[i] is the loss on step i
        corrects = [0. for _ in range(self.K + 1)]
        min_dist_corrects = [0. for _ in range(self.K + 1)]
        # sum_reversed_ranks = [0. for _ in range(self.update_step + 1)]

        granularity = self.granularity
        loss_fn = self.loss_fn

        exp_soft_dropout_power = self.exponential.sample((task_num,)).to(self.device)

        loss_gap = 0

        if self.use_spatial_hierarchy:
            # osm_feature_dict = [task_batch[i]['osm_feature_dict'] for i in range(task_num)]
            # osm_task_feature = torch.stack([osm_feature_dict[i]['osm_task_feature'] for i in range(task_num)])
            # osm_region_avg = torch.stack([osm_feature_dict[i]['osm_region_avg'] for i in range(task_num)])
            # if 'semantic' in self.spatial_hierarchy_info_list:
            #     osm_region_hierarchy_feature = self.hierarchical_activator(osm_region_avg)
            #     _, _, spatial_adv_intensity_list = self.hierarchical_task_cluster(osm_task_feature, osm_region_hierarchy_feature=osm_region_hierarchy_feature)

            semantic_dict = [task_batch[i]['semantic_dict'] for i in range(task_num)]
            task_level_semantics = torch.stack([semantic_dict[i]['task_level_semantics'] for i in range(task_num)])      # task-semantics
            multi_level_semantics = torch.stack([semantic_dict[i]['multi_level_semantics'] for i in range(task_num)])       # region-avg-semantics
            if 'semantic' in self.spatial_hierarchy_info_list:
                gamma = self.lrs(task_level_semantics, multi_level_semantics)

        for i in range(task_num):

            # grad_dirs = []
            # weighted_grad_dirs = []

            tid = task_batch[i]['tid']
            x_spt = task_batch[i]['x_spt']
            y_spt = task_batch[i]['y_spt']
            x_qry = task_batch[i]['x_qry']
            y_qry = task_batch[i]['y_qry']
            # osm_feature_dict = task_batch[i]['osm_feature_dict']

            x_dist_spt = x_spt['x_dist']
            x_local_eid_spt = x_spt['x_local_eid']
            x_global_eid_spt = x_spt['x_global_eid']
            x_dest_dir_spt = x_spt['x_dest_dir']
            data_length_spt = x_spt['data_length']
            y_label_spt = y_spt['y_label']
            nonzero_spt = x_spt['nonzero_spt']

            x_dist_qry = x_qry['x_dist']
            x_local_eid_qry = x_qry['x_local_eid']
            x_global_eid_qry = x_qry['x_global_eid']
            x_dest_dir_qry = x_qry['x_dest_dir']
            data_length_qry = x_qry['data_length']
            y_label_qry = y_qry['y_label']

            # osm_task_feature = osm_feature_dict['osm_task_feature']
            # osm_region_avg = osm_feature_dict['osm_region_avg']

            sample_num_spt = x_dist_spt.size(0)
            sample_num_qry = x_dist_qry.size(0)

            spt_sz += sample_num_spt
            qry_sz += sample_num_qry

            alpha = self.alpha
            if training:
                soft_dropout_num = exp_soft_dropout_power[i] * epoch
                alpha *= (1 - self.eta) ** soft_dropout_num

            # x_raw_osm_hidden_feature, x_enhanced_osm_feature = osm_task_feature, None
            rid_mask = None
            spatial_adv_intensity = 1.
            if self.use_spatial_hierarchy:
                if 'geographical' in self.spatial_hierarchy_info_list:
                    rid_mask = torch.ones(self.level_num).to(self.device)

                if 'semantic' in self.spatial_hierarchy_info_list:
                    row_id = ((tid - 1) // granularity)
                    col_id = ((tid - 1) % granularity)
                    alpha *= gamma[i].squeeze()
                    if training:
                        gamma_dict[row_id][col_id] = gamma[i].item()

            loss_qry_list = torch.tensor([0. for _ in range(self.K + 1)]).to(self.device)
            task_corrects = torch.tensor([0. for _ in range(self.K + 1)]).to(self.device)

            # 0. Query-First
            y_logits_qry, mix_ratio_dict, region_id = self.net(x_dist_qry, x_local_eid_qry, x_global_eid_qry, x_dest_dir_qry,
                                                               tid, granularity, sample_num_qry, data_length_qry,
                                                               rid_mask=rid_mask, params=None, training=training)

            loss_qry = 0
            for t in range(sample_num_qry):
                loss_qry = loss_qry + loss_fn(y_logits_qry[t], y_label_qry[t])
            losses_qry[0] += loss_qry
            loss_qry_list[0] += loss_qry

            with torch.no_grad():
                y_pred_qry = torch.tensor([l_q.argmax() for l_q in y_logits_qry]).to(self.device)
                correct = torch.eq(y_pred_qry, y_label_qry).sum().item()
                corrects[0] = corrects[0] + correct
                task_corrects[0] = task_corrects[0] + correct

                min_pred_q = torch.tensor([torch.argmin(dist[:length]) for (dist, length) in zip(x_dist_qry, data_length_qry)]).to(self.device)
                min_dist_correct = torch.eq(min_pred_q, y_label_qry).sum().item()
                min_dist_corrects[0] = min_dist_corrects[0] + min_dist_correct

            fast_weights = dict(map(lambda p: p, self.net.named_parameters()))
            for k in range(1, self.K + 1):

                # 1. Support-Step-k
                if nonzero_spt:
                    y_logits_spt, _, region_id = self.net(x_dist_spt, x_local_eid_spt, x_global_eid_spt, x_dest_dir_spt,
                                                          tid, granularity, sample_num_spt, data_length_spt,
                                                          rid_mask=rid_mask, params=fast_weights, training=training)

                    loss_spt = 0
                    for t in range(sample_num_spt):
                        loss_spt = loss_spt + loss_fn(y_logits_spt[t], y_label_spt[t])
                        # if self.label_auto_corr:
                        #     loss_spt = loss_spt + F.kl_div(y_logits_spt[t], F.softmax(eid_corr_mat_spt[t][y_label_spt[t]][:data_length_spt[t]]))
                        # label_t = F.one_hot(y_label_spt[t], num_classes=data_length_spt[t]).float()
                        # loss_spt = loss_spt + local_update_loss_fn(y_logits_spt[t], label_t)

                    # hessian = torch.autograd.functional.hessian(loss_spt, tuple([_.view(-1) for _ in self.net.parameters]), create_graph=True)
                    grad = torch.autograd.grad(loss_spt, self.net.parameters(), allow_unused=True, retain_graph=True)
                    # if self.use_dropgrad and training:
                    #     grad = [self.drop_grad(g) for g in grad]
                    # # grad_dir = torch.cat(([g.reshape(1, -1) for g in grad]), dim=1)
                    # # grad_dirs.append(grad_dir)
                    weight = [int(param_name in self.info_param_map.keys()) for (param_name, _) in self.net.named_parameters()]

                    # if training:
                    #     grad_dropout = torch.normal(1., 0.1 / 0.9, (len(self.net.params),)).to(self.device)
                    #     weight = [(int(param_name in self.info_param_map.keys()) * dp) for ((param_name, _), dp) in zip(self.net.named_parameters(), grad_dropout)]
                    # else:
                    #     weight = [int(param_name in self.info_param_map.keys()) for (param_name, _) in self.net.named_parameters()]

                    # weight = [(int(param_name in self.info_param_map.keys()) + 0.1 * int('mixture' in param_name)) for (param_name, _) in self.net.named_parameters()]
                    # weighted_grad_dir = torch.cat(([w * g.reshape(1, -1) for (g, w) in zip(grad, weight)]), dim=1)
                    # weighted_grad_dirs.append(weighted_grad_dir)
                    param_name_list = [name for name in fast_weights.keys()]
                    param_list = [param for param in fast_weights.values()]
                    fast_weights_list = list(map(lambda p: p[1] - p[2] * alpha * p[0], zip(grad, param_list, weight)))
                    fast_weights = dict(map(lambda p: p, zip(param_name_list, fast_weights_list)))

                # 1. Query-Step-k
                y_logits_qry, _, region_id = self.net(x_dist_qry, x_local_eid_qry, x_global_eid_qry, x_dest_dir_qry,
                                                      tid, granularity, sample_num_qry, data_length_qry,
                                                      rid_mask=rid_mask, params=fast_weights, training=training)

                loss_qry = 0
                for t in range(sample_num_qry):
                    loss_qry = loss_qry + loss_fn(y_logits_qry[t], y_label_qry[t])
                    # if self.label_auto_corr:
                    #     loss_qry = loss_qry + F.kl_div(y_logits_qry[t], F.softmax(eid_corr_mat_qry[t][y_label_qry[t]][:data_length_qry[t]]))
                losses_qry[k] += loss_qry
                loss_qry_list[k] += loss_qry

                with torch.no_grad():
                    y_pred_qry = torch.tensor([l_q.argmax() for l_q in y_logits_qry]).to(self.device)
                    correct = torch.eq(y_pred_qry, y_label_qry).sum().item()
                    corrects[k] = corrects[k] + correct
                    task_corrects[k] = task_corrects[k] + correct

                    min_pred_q = torch.tensor([torch.argmin(dist[:length]) for (dist, length) in zip(x_dist_qry, data_length_qry)]).to(self.device)
                    min_dist_correct = torch.eq(min_pred_q, y_label_qry).sum().item()
                    min_dist_corrects[k] = min_dist_corrects[k] + min_dist_correct

        batch_loss = losses_qry[-1]
        batch_loss = batch_loss / task_num

        return min_dist_corrects, corrects, batch_loss, qry_sz, gamma_dict