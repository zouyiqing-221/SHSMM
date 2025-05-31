import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class LRS_Semantic_Clustering(nn.Module):
    def __init__(self, cluster_tree_hierarchy_structure=None, hidden_dim=64):
        super(LRS_Semantic_Clustering, self).__init__()

        if cluster_tree_hierarchy_structure is None:
            cluster_tree_hierarchy_structure = [4, 4, 4, 1]

        self.cluster_tree_hierarchy_structure = cluster_tree_hierarchy_structure
        self.hidden_dim = hidden_dim

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.params = nn.ParameterDict()

        for i, N in enumerate(self.cluster_tree_hierarchy_structure):

            if i == 0 and N != 1:
                self.params[f'layer_{i}_leaf_centre'] = \
                    nn.Parameter(nn.init.kaiming_normal_(torch.randn(N, self.hidden_dim).to(device)), requires_grad=True)
                for n in range(N):
                    self.params[f'layer_{i}_leaf_weight_{n}'] = \
                        nn.Parameter(nn.init.kaiming_normal_(torch.randn(self.hidden_dim, self.hidden_dim).to(device)), requires_grad=True)
                    self.params[f'layer_{i}_leaf_bias_{n}'] = \
                        nn.Parameter(nn.init.kaiming_normal_(torch.randn(1, self.hidden_dim).to(device)), requires_grad=True)
            elif i != 0 and N != 1:
                self.params[f'layer_{i}_mid_centre'] = \
                    nn.Parameter(nn.init.kaiming_normal_(torch.randn(N, self.hidden_dim).to(device)), requires_grad=True)
                for n in range(N):
                    self.params[f'layer_{i}_mid_weight_{n}'] = \
                        nn.Parameter(nn.init.kaiming_normal_(torch.randn(self.hidden_dim, self.hidden_dim).to(device)), requires_grad=True)
                    self.params[f'layer_{i}_mid_bias_{n}'] = \
                        nn.Parameter(nn.init.kaiming_normal_(torch.randn(1, self.hidden_dim).to(device)), requires_grad=True)
            elif i != 0 and N == 1:
                self.params[f'layer_{i}_root_centre'] = \
                    nn.Parameter(nn.init.kaiming_normal_(torch.randn(N, self.hidden_dim).to(device)), requires_grad=True)
                self.params[f'layer_{i}_root_weight'] = \
                    nn.Parameter(nn.init.kaiming_normal_(torch.randn(self.hidden_dim, self.hidden_dim).to(device)), requires_grad=True)
                self.params[f'layer_{i}_root_bias'] = \
                    nn.Parameter(nn.init.kaiming_normal_(torch.randn(1, self.hidden_dim).to(device)), requires_grad=True)
                break
            else:
                self.params[f'unique_centre'] = \
                    nn.Parameter(nn.init.kaiming_normal_(torch.randn(self.hidden_dim, self.osm_feature_dim).to(device)), requires_grad=True)
                self.params[f'layer_{i}_root_weight'] = \
                    nn.Parameter(nn.init.kaiming_normal_(torch.randn(self.hidden_dim, self.osm_feature_dim).to(device)), requires_grad=True)
                self.params[f'layer_{i}_root_bias'] = \
                    nn.Parameter(nn.init.kaiming_normal_(torch.randn(1, self.hidden_dim).to(device)), requires_grad=True)
                break

    def forward(self, raw_semantic_feature):

        params = self.params

        for i, N in enumerate(self.cluster_tree_hierarchy_structure):

            if i == 0 and N != 1:
                cluster_center = params[f'layer_{i}_leaf_centre']
                soft_assign_rate = torch.softmax(- torch.norm(raw_semantic_feature.unsqueeze(1) - cluster_center, dim=-1, p=2), dim=-1)
                hidden_semantic_feature = soft_assign_rate.unsqueeze(-1) * \
                                     torch.stack([torch.tanh(F.linear(raw_semantic_feature,
                                                                      params[f'layer_{i}_leaf_weight_{n}'],
                                                                      params[f'layer_{i}_leaf_bias_{n}'].squeeze())) for n in range(N)], dim=1)
                # layer_assign_rate.append(soft_assign_rate.squeeze().detach().numpy())
            elif i != 0 and N != 1:
                mid_cluster_center = params[f'layer_{i}_mid_centre']
                mid_soft_assign_rate = [torch.softmax(- torch.norm(hidden_semantic_feature[:, n:n+1, :] - mid_cluster_center, dim=-1, p=2), dim=-1)
                                        for n in range(self.cluster_tree_hierarchy_structure[i - 1])]
                mid_soft_assign_rate = torch.stack(mid_soft_assign_rate, dim=1).unsqueeze(-1)
                hidden_semantic_feature = torch.stack([torch.tanh(F.linear(hidden_semantic_feature,
                                                                      params[f'layer_{i}_mid_weight_{n}'],
                                                                      params[f'layer_{i}_mid_bias_{n}'].squeeze())).squeeze() for n in range(N)], dim=-2)
                hidden_semantic_feature = torch.sum(mid_soft_assign_rate * hidden_semantic_feature, dim=1)
                # soft_assign_rate = (soft_assign_rate * mid_soft_assign_rate.squeeze(-1).T).sum(dim=0).unsqueeze(-1)
                # layer_assign_rate.append(soft_assign_rate.squeeze().detach().numpy())
            elif i != 0 and N == 1:
                root_cluster_center = params[f'layer_{i}_root_centre']
                root_soft_assign_rate = [torch.softmax(- torch.norm(hidden_semantic_feature[:, n:n+1, :] - root_cluster_center, dim=-1, p=2), dim=-1)
                                         for n in range(self.cluster_tree_hierarchy_structure[i - 1])]
                root_soft_assign_rate = torch.stack(root_soft_assign_rate, dim=1).unsqueeze(-1)
                hidden_semantic_feature = torch.tanh(F.linear(hidden_semantic_feature,
                                                         params[f'layer_{i}_root_weight'],
                                                         params[f'layer_{i}_root_bias'].squeeze())).unsqueeze(-2)
                enhanced_semantic_feature = torch.sum(root_soft_assign_rate * hidden_semantic_feature, dim=1)
                # soft_assign_rate = (soft_assign_rate * root_soft_assign_rate.squeeze(-1).T).sum(dim=0).unsqueeze(-1)
                # layer_assign_rate.append(soft_assign_rate.broadcast_to((4, 1)).squeeze().detach().numpy())
                break

        return enhanced_semantic_feature

    def parameters(self, recurse: bool = False):

        return self.params.values()

    def named_parameters(self, prefix: str = '', recurse: bool = True):

        return self.params.items()


class LRS_Cross_Level_Feature_Enhancement(nn.Module):
    def __init__(self, single_level_feature_dim=1000, hidden_dim=100):
        super(LRS_Cross_Level_Feature_Enhancement, self).__init__()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.params = nn.ParameterDict()
        self.params['osm_emb_weight'] = nn.Parameter(nn.init.kaiming_normal_(torch.randn(hidden_dim, single_level_feature_dim).to(device)),
                                                     requires_grad=True)
        self.params['osm_emb_bias'] = nn.Parameter(nn.init.kaiming_normal_(torch.randn(1, hidden_dim).to(device)),
                                                   requires_grad=True)

    def forward(self, multi_level_semantics):
        params = self.params
        batch_size = multi_level_semantics.size(0)
        multi_level_semantics_enc = F.linear(multi_level_semantics, params['osm_emb_weight'], params['osm_emb_bias'].squeeze()).squeeze()
        semantic_hierarchy_corr = torch.softmax(torch.matmul(multi_level_semantics_enc, multi_level_semantics_enc.transpose(-1, -2)), dim=-1)
        hierarchical_semantics = torch.matmul(semantic_hierarchy_corr, multi_level_semantics_enc).reshape(batch_size, -1)

        return hierarchical_semantics

    def parameters(self, recurse: bool = False):

        return self.params.values()

    def named_parameters(self, prefix: str = '', recurse: bool = True):

        return self.params.items()


class LRS_LR_Scaling_Factor_Generator(nn.Module):
    def __init__(self, single_level_feature_dim=1000, hidden_dim=64, cluster_tree_hierarchy_structure=None,
                 sigma=20.0, activation = 'sech'):
        super(LRS_LR_Scaling_Factor_Generator, self).__init__()

        if cluster_tree_hierarchy_structure is None:
            cluster_tree_hierarchy_structure = [64, 16, 4, 1]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.sigma = sigma
        self.activation = activation

        self.params = nn.ParameterDict()

        self.params['task_osm_emb_weight'] = \
            nn.Parameter(nn.init.kaiming_normal_(torch.randn(hidden_dim, single_level_feature_dim).to(device)), requires_grad=True)
        self.params['task_osm_emb_bias'] = \
            nn.Parameter(nn.init.kaiming_normal_(torch.randn(1, hidden_dim).to(device)), requires_grad=True)

        self.hierarchical_soft_cluster = LRS_Semantic_Clustering(cluster_tree_hierarchy_structure=cluster_tree_hierarchy_structure,
                                                                 hidden_dim=hidden_dim).to(device)

        self.params['task_osm_emb_to_scalar'] = \
            nn.Parameter(nn.init.kaiming_normal_(torch.randn(1, hidden_dim * 2).to(device)), requires_grad=True)

    def forward(self, task_level_semantics, hierarchical_semantics=None):

        params = self.params

        task_semantic_feture = F.linear(task_level_semantics, params['task_osm_emb_weight'], params['task_osm_emb_bias'].squeeze()).unsqueeze(1)
        if hierarchical_semantics is None:
            raw_semantic_feature = F.linear(task_level_semantics, params['task_osm_emb_weight'], params['task_osm_emb_bias'].squeeze())
        else:
            raw_semantic_feature = F.linear(hierarchical_semantics, params['task_osm_emb_weight'], params['task_osm_emb_bias'].squeeze())

        enhanced_semantic_feature = self.hierarchical_soft_cluster(raw_semantic_feature)

        raw_scaling_factor = F.linear(torch.cat((task_semantic_feture, enhanced_semantic_feature), dim=-1), params['task_osm_emb_to_scalar'])
        if self.activation == 'sech':
            task_lr_scaling_factor = 2 / (torch.exp(raw_scaling_factor / self.sigma) + torch.exp(- raw_scaling_factor / self.sigma))
        elif self.activation == 'sigmoid':
            task_lr_scaling_factor = torch.sigmoid(raw_scaling_factor / self.sigma)
        else:
            raise Exception('No such activation function')

        return task_lr_scaling_factor

    def parameters(self, recurse: bool = False):

        return self.params.values()

    def named_parameters(self, prefix: str = '', recurse: bool = True):

        return self.params.items()


class Learning_Rate_Scaling(nn.Module):
    def __init__(self,
                 single_level_feature_dim=1000, per_hierarchy_dim=100,
                 hidden_dim=64, cluster_tree_hierarchy_structure=None, enhanced=True,
                 sigma=20.0, activation = 'sech'):
        super(Learning_Rate_Scaling, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.enhanced = enhanced

        if enhanced:
            self.cross_level_feature_enhancement = LRS_Cross_Level_Feature_Enhancement(single_level_feature_dim=single_level_feature_dim,
                                                                                       hidden_dim=per_hierarchy_dim).to(self.device)
        self.get_lr_scaling_factor = LRS_LR_Scaling_Factor_Generator(single_level_feature_dim=single_level_feature_dim,
                                                                     hidden_dim=hidden_dim,
                                                                     cluster_tree_hierarchy_structure=cluster_tree_hierarchy_structure,
                                                                     sigma=sigma, activation=activation).to(self.device)

    def forward(self, task_level_semantics, multi_level_semantics):
        if self.enhanced:
            hierarchical_semantics = self.cross_level_feature_enhancement(multi_level_semantics)
        else:
            hierarchical_semantics = None
        task_lr_scaling_factor = self.get_lr_scaling_factor(task_level_semantics, hierarchical_semantics=hierarchical_semantics)

        return task_lr_scaling_factor


# class Initial_Parameter_Modulation(nn.Module):
#     def __init__(self):
#         super(Initial_Parameter_Modulation, self).__init__()
#
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SHSMM(nn.Module):
    def __init__(self,
                 lid_in_dim=80, lid_emb_dim=1, gid_in_dim=8132, gid_emb_dim=16,
                 transformer_in_dim=64, transformer_hidden_dim=64, num_heads=1,
                 fc_combine_dim=16,
                 level_num=10, rid_emb_dim=3, task_rep_dim=64,
                 use_gk=True,
                 use_spatial_hierarchy=True, spatial_hierarchy_info_list=None,
                 input_cat_destination=False, granularity=512):
        super(SHSMM, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.params = nn.ParameterDict()

        self.transformer_in_dim = transformer_in_dim
        self.transformer_hidden_dim = transformer_hidden_dim
        self.num_heads = num_heads
        self.level_num = level_num
        self.rid_emb_dim = rid_emb_dim

        # self.is_meta = is_meta
        self.use_gk = use_gk
        self.use_spatial_hierarchy = use_spatial_hierarchy
        self.spatial_hierarchy_info_list = spatial_hierarchy_info_list
        # self.spatial_info_fusion_mode = spatial_info_fusion_mode
        # self.mixture_head_num = mixture_head_num

        # self.input_cat_osm = input_cat_osm
        self.input_cat_destination = input_cat_destination
        # self.input_cat_destination_global = input_cat_destination_global
        # self.input_cat_destination_rescale = input_cat_destination_rescale

        # self.rid_type = rid_type
        self.garnularity = granularity

        self.params['edge_emb_w'] = nn.Parameter(nn.init.kaiming_normal_(torch.randn(lid_in_dim, lid_emb_dim)), requires_grad=True)
        if self.input_cat_destination:
            self.params['fc_in_w'] = nn.Parameter(nn.init.kaiming_normal_(torch.randn(transformer_in_dim, 2 + lid_emb_dim)), requires_grad=True)
        else:
            self.params['fc_in_w'] = nn.Parameter(nn.init.kaiming_normal_(torch.randn(transformer_in_dim, 1 + lid_emb_dim)), requires_grad=True)
        self.params['fc_in_b'] = nn.Parameter(nn.init.kaiming_normal_(torch.randn(1, transformer_in_dim)), requires_grad=True)

        self.params['self_attn_in_proj_w'] = nn.Parameter(nn.init.kaiming_normal_(torch.randn(transformer_hidden_dim * 3, transformer_hidden_dim)),
                                                          requires_grad=True)
        self.params['self_attn_in_proj_b'] = nn.Parameter(nn.init.kaiming_normal_(torch.randn(1, transformer_hidden_dim * 3)),
                                                          requires_grad=True)
        self.params['self_attn_out_proj_w'] = nn.Parameter(nn.init.kaiming_normal_(torch.randn(transformer_hidden_dim, transformer_hidden_dim)),
                                                           requires_grad=True)
        self.params['self_attn_out_proj_b'] = nn.Parameter(nn.init.kaiming_normal_(torch.randn(1, transformer_hidden_dim)),
                                                           requires_grad=True)
        self.params['linear1_w'] = nn.Parameter(nn.init.kaiming_normal_(torch.randn(transformer_hidden_dim, transformer_hidden_dim)),
                                                requires_grad=True)
        self.params['linear1_b'] = nn.Parameter(nn.init.kaiming_normal_(torch.randn(1, transformer_hidden_dim)),
                                                requires_grad=True)
        self.params['linear2_w'] = nn.Parameter(nn.init.kaiming_normal_(torch.randn(transformer_hidden_dim, transformer_hidden_dim)),
                                                requires_grad=True)
        self.params['linear2_b'] = nn.Parameter(nn.init.kaiming_normal_(torch.randn(1, transformer_hidden_dim)),
                                                requires_grad=True)

        if self.use_gk:
            self.params['global_edge_emb_w'] = nn.Parameter(nn.init.kaiming_normal_(torch.randn(gid_in_dim, gid_emb_dim)), requires_grad=True)
            self.params['fc_out_w'] = nn.Parameter(nn.init.kaiming_normal_(torch.randn(fc_combine_dim, transformer_hidden_dim + gid_emb_dim)),
                                                       requires_grad=True)
        else:
            self.params['fc_out_w'] = nn.Parameter(nn.init.kaiming_normal_(torch.randn(fc_combine_dim, transformer_hidden_dim)), requires_grad=True)
        self.params['fc_combine_w'] = nn.Parameter(nn.init.kaiming_normal_(torch.randn(1, fc_combine_dim)), requires_grad=True)

        if self.use_spatial_hierarchy and 'geographical' in self.spatial_hierarchy_info_list:
            self.params['task_edge_emb_w'] = nn.Parameter(nn.init.kaiming_normal_(torch.randn(lid_in_dim * lid_emb_dim, task_rep_dim)), requires_grad=True)
            if self.input_cat_destination:
                self.params['task_fc_in_w'] = nn.Parameter(nn.init.kaiming_normal_(torch.randn(transformer_in_dim * (2 + lid_emb_dim),
                                                                                               task_rep_dim)), requires_grad=True)
            else:
                self.params['task_fc_in_w'] = nn.Parameter(nn.init.kaiming_normal_(torch.randn(transformer_in_dim * (1 + lid_emb_dim), task_rep_dim)),
                                                           requires_grad=True)
            self.params['task_fc_in_b'] = nn.Parameter(nn.init.kaiming_normal_(torch.randn(transformer_in_dim, task_rep_dim)), requires_grad=True)

            self.params['task_self_attn_in_proj_w'] = nn.Parameter(nn.init.kaiming_normal_(torch.randn(transformer_hidden_dim * 3 * transformer_hidden_dim,
                                                                                                       task_rep_dim)),
                                                                   requires_grad=True)
            self.params['task_self_attn_in_proj_b'] = nn.Parameter(nn.init.kaiming_normal_(torch.randn(transformer_hidden_dim * 3, task_rep_dim)),
                                                                   requires_grad=True)
            self.params['task_self_attn_out_proj_w'] = nn.Parameter(nn.init.kaiming_normal_(torch.randn(transformer_hidden_dim * transformer_hidden_dim,
                                                                                                        task_rep_dim)),
                                                                    requires_grad=True)
            self.params['task_self_attn_out_proj_b'] = nn.Parameter(nn.init.kaiming_normal_(torch.randn(transformer_hidden_dim, task_rep_dim)),
                                                                    requires_grad=True)
            self.params['task_linear1_w'] = nn.Parameter(nn.init.kaiming_normal_(torch.randn(transformer_hidden_dim * transformer_hidden_dim,
                                                                                             task_rep_dim)),
                                                         requires_grad=True)
            self.params['task_linear1_b'] = nn.Parameter(nn.init.kaiming_normal_(torch.randn(transformer_hidden_dim, task_rep_dim)),
                                                         requires_grad=True)
            self.params['task_linear2_w'] = nn.Parameter(nn.init.kaiming_normal_(torch.randn(transformer_hidden_dim * transformer_hidden_dim,
                                                                                             task_rep_dim)),
                                                         requires_grad=True)
            self.params['task_linear2_b'] = nn.Parameter(nn.init.kaiming_normal_(torch.randn(transformer_hidden_dim, task_rep_dim)),
                                                         requires_grad=True)

            if self.use_gk:
                self.params['task_fc_out_w'] = nn.Parameter(nn.init.kaiming_normal_(torch.randn(fc_combine_dim * (transformer_hidden_dim + gid_emb_dim),
                                                                                                    task_rep_dim)), requires_grad=True)
                self.params['task_fc_combine_w'] = nn.Parameter(nn.init.kaiming_normal_(torch.randn(fc_combine_dim, task_rep_dim)), requires_grad=True)
            else:
                self.params['task_fc_out_w'] = nn.Parameter(nn.init.kaiming_normal_(torch.randn(fc_combine_dim * transformer_hidden_dim, task_rep_dim)),
                                                                requires_grad=True)
                self.params['task_fc_combine_w'] = nn.Parameter(nn.init.kaiming_normal_(torch.randn(fc_combine_dim, task_rep_dim)), requires_grad=True)

            self.params['rid_emb_w'] = nn.Parameter(nn.init.kaiming_normal_(torch.randn(4, rid_emb_dim)), requires_grad=True)
            self.params['task_rprsnt_w'] = nn.Parameter(nn.init.kaiming_normal_(torch.randn(task_rep_dim, rid_emb_dim * level_num)), requires_grad=True)
            self.params['task_rprsnt_b'] = nn.Parameter(nn.init.kaiming_normal_(torch.randn(1, task_rep_dim)), requires_grad=True)

    def forward(self, x_dist, x_local_eid, x_global_eid, x_dest_dir,
                tid=None, granularity=None, sample_num=None, data_length=None, rid_mask=None,
                params=None, training=False):
        """
        Args:
            x_dist: projection distances from gps point to candidate roads.
            x_local_eid: corresponding road id within the specific task.
            x_global_eid: corresponding road id within the whole region.
            x_osm_feature: 1000-dim representation of the task comes from open-street-map.
            tid: task id.
            granularity: granularity of the task.
            sample_num: number of samples in the batch / task.
            params: parameters used in the model.
        """

        mix_ratio_dict = {}

        if params is None:
            params = self.params

        region_id = None

        if self.use_spatial_hierarchy:

            task_rep = None

            if 'geographical' in self.spatial_hierarchy_info_list:
                hierarchical_target_granularity_list, pooling_target_granularity = self.get_target_granularities(level_num=self.level_num)
                quad_tid_list = self.quadtree_hierarchy_region_id(task_id=tid,
                                                                  raw_granularity=granularity,
                                                                  target_granularity_list=hierarchical_target_granularity_list)
                region_id = torch.tensor(quad_tid_list, dtype=torch.long).to(self.device)
                rid_emb = F.embedding(region_id, params['rid_emb_w'])
                rid_emb = (rid_emb * (rid_mask * torch.ones(self.rid_emb_dim, self.level_num).to(self.device)).T).reshape(1, -1)
                task_rep = torch.sigmoid(F.linear(rid_emb, params['task_rprsnt_w'], params['task_rprsnt_b']))

        task_params = {}
        for key in self.params.keys():
            if f'task_{key}' in self.params.keys():
                # param_size = len(self.var_dict[key].view(-1))
                if self.use_spatial_hierarchy:
                    size_list = self.params[key].shape
                    size_tuple = []
                    for i in range(size_list.__len__()):
                        size_tuple.append(size_list[i])
                    size_tuple = tuple(size_tuple)

                    task_params[key] = torch.sigmoid(F.linear(task_rep, params[f'task_{key}'])).reshape(size_tuple)
                    task_params[key] = params[key] * task_params[key]
                else:
                    task_params[key] = params[key]
            else:
                task_params[key] = params[key]

        local_eid_emb = F.embedding(x_local_eid.squeeze(-1), task_params['edge_emb_w'])
        if self.input_cat_destination:
            transformer_input = torch.cat((x_dist, local_eid_emb, x_dest_dir), dim=2)
            transformer_input = F.linear(transformer_input, task_params['fc_in_w'], task_params['fc_in_b'])
        else:
            transformer_input = torch.cat((x_dist, local_eid_emb), dim=2)
            transformer_input = F.linear(transformer_input, task_params['fc_in_w'], task_params['fc_in_b'])

        x = transformer_input.permute(1, 0, 2)
        y1, _ = F.multi_head_attention_forward(x, x, x, self.transformer_hidden_dim, self.num_heads,
                                               in_proj_weight=task_params['self_attn_in_proj_w'],
                                               in_proj_bias=task_params['self_attn_in_proj_b'],
                                               bias_k=None, bias_v=None,
                                               out_proj_weight=task_params['self_attn_out_proj_w'],
                                               out_proj_bias=task_params['self_attn_out_proj_b'],
                                               add_zero_attn=False, dropout_p=0.1, training=training, need_weights=False)
        y1 = F.layer_norm(x + F.dropout(y1, p=0.1, training=training), [self.transformer_hidden_dim])
        y2 = F.linear(F.dropout(F.relu(F.linear(y1, task_params['linear1_w'], task_params['linear1_b'])), p=0.1, training=training), task_params['linear2_w'], task_params['linear2_b'])
        transformer_out = F.layer_norm(y1 + F.dropout(y2, p=0.1, training=training), [self.transformer_hidden_dim]).permute(1, 0, 2)

        if self.use_gk is True:
            global_knowledge = F.embedding(x_global_eid.squeeze(-1), params['global_edge_emb_w'])
            transformer_out = torch.cat((transformer_out, global_knowledge), dim=-1)

        y_pred = []
        for i in range(sample_num):
            linear_out_i = F.linear(transformer_out[i][:data_length[i]], task_params['fc_out_w'])
            combine_out_i = F.linear(torch.tanh(linear_out_i), task_params['fc_combine_w']).squeeze(dim=1)
            y_pred_i = F.log_softmax(combine_out_i, dim=0)
            y_pred.append(y_pred_i)

        return y_pred, mix_ratio_dict, region_id

    def get_target_granularities(self, level_num):

        pooling_target_granularity = 2 ** (level_num - 1)

        hierarchical_target_granularity_list = []
        for l in range(level_num):
            target_g = 2 ** l
            hierarchical_target_granularity_list.append(target_g)

        return hierarchical_target_granularity_list, pooling_target_granularity

    def quadtree_hierarchy_region_id(self, task_id, raw_granularity, target_granularity_list):

        target_quad_tid_list = []
        for g in target_granularity_list:
            row_id = ((task_id - 1) // raw_granularity) // (raw_granularity / g)
            col_id = ((task_id - 1) % raw_granularity) // (raw_granularity / g)
            qtid = (row_id % 2) * 2 + (col_id % 2)
            target_quad_tid_list.append(qtid)

        return target_quad_tid_list

    def get_row_col_id_by_tid(self, task_id, raw_granularity, target_granularity):
        row_id = ((task_id - 1) // raw_granularity) // (raw_granularity / target_granularity)
        col_id = ((task_id - 1) % raw_granularity) // (raw_granularity / target_granularity)
        rid = row_id * target_granularity + col_id
        return row_id, col_id, rid

    def parameters(self, recurse: bool = False):

        return self.params.values()

    def named_parameters(self, prefix: str = '', recurse: bool = True):

        return self.params.items()