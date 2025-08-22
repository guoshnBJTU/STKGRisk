import genericpath
from glob import glob
from tokenize import group
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import sys
import pandas as pd
import time
from datetime import timedelta
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from model.encoder import *
from model.MMOE import *
from lib.utils import mix_zip_nll_loss

class Hypernet(nn.Module):
    """
        Hypernetwork deals with decoder input and generates params for mu, sigma, w
    Args:
        config: Model configuration.
        hidden_sizes: Sizes of the hidden layers. [] corresponds to a linear layer.
        param_sizes: Sizes of the output parameters. [n_components, n_components, n_components]
        activation: Activation function.
    """
    def __init__(self, input_size, hidden_sizes=[], param_sizes=[1, 1], activation=nn.Tanh()):
        super().__init__()
        self.input_size = input_size
        self.activation = activation

        ends = torch.cumsum(torch.tensor(param_sizes), dim=0)
        starts = torch.cat((torch.zeros(1).type_as(ends), ends[:-1]))
        self.param_slices = [slice(s.item(), e.item()) for s, e in zip(starts, ends)]

        self.output_size = sum(param_sizes)
        layer_sizes = list(hidden_sizes) + [self.output_size]

        self.first_bias = nn.Parameter(torch.empty(layer_sizes[0]).uniform_(-0.1, 0.1))
        self.first_linear = nn.Linear(self.input_size, layer_sizes[0], bias=False)

        self.linear_layers = nn.ModuleList()
        for idx, size in enumerate(layer_sizes[:-1]):
            self.linear_layers.append(nn.Linear(size, layer_sizes[idx + 1]))
        
        self.reset_parameters()

    def forward(self, input):
        """Generate model parameters from the embeddings.

        Args:
            input: decoder input, shape (batch, input_size)

        Returns:
            params: Tuple of model parameters.
        """
        hidden = self.first_bias
        hidden = hidden + self.first_linear(input)
        for layer in self.linear_layers:
            hidden = layer(self.activation(hidden))

        return tuple([hidden[..., s] for s in self.param_slices])
    
    def reset_parameters(self):
        print("use normal_")
        nn.init.normal_(self.first_linear.weight)
        for layer in self.linear_layers:
            nn.init.normal_(layer.weight)


class GCN_Layer(nn.Module):
    def __init__(self, num_of_features, num_of_filter, depth=False):
        """One layer of GCN
        
        Arguments:
            num_of_features {int} -- the dimension of node feature
            num_of_filter {int} -- the number of graph filters
        """
        super(GCN_Layer, self).__init__()
        self.gcn_layer = nn.Sequential(
            nn.Linear(in_features=num_of_features, out_features=num_of_filter), 
            nn.ReLU()
        )
        self.depth = depth

    def forward(self, input, adj): # AXW
        """
        Arguments:
            input {Tensor} -- signal matrix, shape (batch_size, N, T * D) -> 修正为 (batch_size * T, N, D)
            adj {np.array} -- adjacent matrix，shape (N, N)
        Returns:
            {Tensor} -- output, shape (batch_size, N, num_of_filter)
        """
        batch_size, _, _ = input.shape
        # print("gcn", batch_size) # batch_size * T
        
        if self.depth is False:
            adj = torch.from_numpy(adj).to(input.device).to(torch.float32)
        adj = adj.repeat(batch_size, 1, 1)
        input = torch.bmm(adj, input)
        output = self.gcn_layer(input)
        return output

class STGN(nn.Module):
    def __init__(self, num_of_graph_feature, nums_of_graph_filters, 
                seq_len, num_of_gru_layers, gru_hidden_size, 
                num_of_target_time_feature, num_of_graph_node, prefer_depth=False, zone2seg=None,
                depth_gcn=False, pretrain_kg=False, depth_adjs_path=None, kg_model_path=None,
                encoder_layer=3, dropout=0.1, n_head=4, node_emb_size=100):
        """
        Arguments:
            num_of_graph_feature {int} -- the number of graph node feature，(batch_size, seq_len, D, N), num_of_graph_feature=D
            nums_of_graph_filters {list} -- the number of GCN output feature
            seq_len {int} -- the time length of input
            num_of_gru_layers {int} -- the number of GRU layers
            gru_hidden_size {int} -- the hidden size of GRU
            num_of_target_time_feature {int} -- the number of target time feature，为 24(hour) + 7(week) + 1(holiday) = 32
        """
        super(STGN, self).__init__()
        self.road_gcn_seg = nn.ModuleList()

        self.num_of_graph_node = num_of_graph_node
        self.zone2seg = zone2seg.unsqueeze(dim=0) # (1, 69, 475)

        for idx, num_of_filter in enumerate(nums_of_graph_filters):
            if idx == 0:
                self.road_gcn_seg.append(GCN_Layer(num_of_graph_feature, num_of_filter))
            else:
                self.road_gcn_seg.append(GCN_Layer(nums_of_graph_filters[idx - 1], num_of_filter))
        self.zone_gcn = nn.ModuleList()
        for idx, num_of_filter in enumerate(nums_of_graph_filters):
            if idx == 0:
                self.zone_gcn.append(GCN_Layer(num_of_graph_feature, num_of_filter))
            else:
                self.zone_gcn.append(GCN_Layer(nums_of_graph_filters[idx - 1], num_of_filter))
        
        self.risk_gcn_seg = nn.ModuleList()
        for idx, num_of_filter in enumerate(nums_of_graph_filters):
            if idx == 0:
                self.risk_gcn_seg.append(GCN_Layer(num_of_graph_feature, num_of_filter))
            else:
                self.risk_gcn_seg.append(GCN_Layer(nums_of_graph_filters[idx - 1], num_of_filter))
        self.risk_gcn_zone = nn.ModuleList()
        for idx, num_of_filter in enumerate(nums_of_graph_filters):
            if idx == 0:
                self.risk_gcn_zone.append(GCN_Layer(num_of_graph_feature, num_of_filter))
            else:
                self.risk_gcn_zone.append(GCN_Layer(nums_of_graph_filters[idx - 1], num_of_filter))

        self.poi_gcn_seg = nn.ModuleList()
        for idx, num_of_filter in enumerate(nums_of_graph_filters):
            if idx == 0:
                self.poi_gcn_seg.append(GCN_Layer(num_of_graph_feature, num_of_filter))
            else:
                self.poi_gcn_seg.append(GCN_Layer(nums_of_graph_filters[idx - 1], num_of_filter))
        self.poi_gcn_zone = nn.ModuleList()
        for idx, num_of_filter in enumerate(nums_of_graph_filters):
            if idx == 0:
                self.poi_gcn_zone.append(GCN_Layer(num_of_graph_feature, num_of_filter))
            else:
                self.poi_gcn_zone.append(GCN_Layer(nums_of_graph_filters[idx - 1], num_of_filter))

        # 自适应图卷积
        self.depth_gcn = depth_gcn
        self.pretrain_kg = pretrain_kg
        self.node_emb_size = node_emb_size
        if depth_gcn is True:
            self.softmax = nn.Softmax(dim=1)
            self.prefer_depth = prefer_depth

            if pretrain_kg:
                kg_model = torch.load(kg_model_path, map_location=torch.device(f"cuda:0"))
                if isinstance(kg_model, nn.DataParallel):
                    kg_model = kg_model.module
                self.de_simple = kg_model # 微调？
                
                depth_adjs = np.load(depth_adjs_path, allow_pickle=True)
                self.seg_poi_adj = torch.from_numpy(depth_adjs["seg_poi_adj"]).float()
                self.seg_rf_adj = torch.from_numpy(depth_adjs["seg_rf_adj"]).float()
                self.zone_poi_adj = torch.from_numpy(depth_adjs["zone_include_poi_adj"]).float()
                self.zone_rf_adj = torch.from_numpy(depth_adjs["zone_rf_adj"]).float()
                self.seg_dict = depth_adjs["seg_dict"].item()
                self.poi_dict = depth_adjs["poi_dict"].item()
                self.rf_dict = depth_adjs["rf_dict"].item()
                self.zone_dict = depth_adjs["zone_dict"].item()
            else:
                self.seg_emb = nn.Embedding(num_of_graph_node[0], node_emb_size)
                self.risk_emb_seg = nn.Embedding(num_of_graph_node[0], node_emb_size)
                self.poi_emb_seg = nn.Embedding(num_of_graph_node[0], node_emb_size)
                self.zone_emb = nn.Embedding(num_of_graph_node[1], node_emb_size)
                self.risk_emb_zone = nn.Embedding(num_of_graph_node[1], node_emb_size)
                self.poi_emb_zone = nn.Embedding(num_of_graph_node[1], node_emb_size)

            self.depth_seg_gcn = nn.ModuleList()
            for idx, num_of_filter in enumerate(nums_of_graph_filters):
                if idx == 0:
                    self.depth_seg_gcn.append(GCN_Layer(num_of_graph_feature, num_of_filter, depth=True))
                else:
                    self.depth_seg_gcn.append(GCN_Layer(nums_of_graph_filters[idx - 1], num_of_filter, depth=True))
            self.depth_zone_gcn = nn.ModuleList()
            for idx, num_of_filter in enumerate(nums_of_graph_filters):
                if idx == 0:
                    self.depth_zone_gcn.append(GCN_Layer(num_of_graph_feature, num_of_filter, depth=True))
                else:
                    self.depth_zone_gcn.append(GCN_Layer(nums_of_graph_filters[idx - 1], num_of_filter, depth=True))

            self.depth_poi_gcn_seg = nn.ModuleList()
            for idx, num_of_filter in enumerate(nums_of_graph_filters):
                if idx == 0:
                    self.depth_poi_gcn_seg.append(GCN_Layer(num_of_graph_feature, num_of_filter, depth=True))
                else:
                    self.depth_poi_gcn_seg.append(GCN_Layer(nums_of_graph_filters[idx - 1], num_of_filter, depth=True))
            self.depth_poi_gcn_zone = nn.ModuleList()
            for idx, num_of_filter in enumerate(nums_of_graph_filters):
                if idx == 0:
                    self.depth_poi_gcn_zone.append(GCN_Layer(num_of_graph_feature, num_of_filter, depth=True))
                else:
                    self.depth_poi_gcn_zone.append(GCN_Layer(nums_of_graph_filters[idx - 1], num_of_filter, depth=True))
            
            self.depth_risk_gcn_seg = nn.ModuleList()
            for idx, num_of_filter in enumerate(nums_of_graph_filters):
                if idx == 0:
                    self.depth_risk_gcn_seg.append(GCN_Layer(num_of_graph_feature, num_of_filter, depth=True))
                else:
                    self.depth_risk_gcn_seg.append(GCN_Layer(nums_of_graph_filters[idx - 1], num_of_filter, depth=True))
            self.depth_risk_gcn_zone = nn.ModuleList()
            for idx, num_of_filter in enumerate(nums_of_graph_filters):
                if idx == 0:
                    self.depth_risk_gcn_zone.append(GCN_Layer(num_of_graph_feature, num_of_filter, depth=True))
                else:
                    self.depth_risk_gcn_zone.append(GCN_Layer(nums_of_graph_filters[idx - 1], num_of_filter, depth=True))

            self.gcn_weights_seg = nn.Parameter(torch.tensor([0.5, 0.5, 0.5]))
            self.gcn_weights_zone = nn.Parameter(torch.tensor([0.5, 0.5, 0.5]))

        self.d_model = num_of_filter * 3

        self.encoder_layer = encoder_layer
        if self.d_model % n_head != 0:
            self.n_head = 2 * 3
        else:
            self.n_head = n_head

        self.attention_seg = MultiHeadedAttention(h=self.n_head, d_model=self.d_model, dropout=dropout)
        self.feed_forward_seg = PositionwiseFeedForward(d_model=self.d_model, d_ff=self.d_model * 4, dropout=dropout)
        self.input_sublayer_seg = SublayerConnection(size=self.d_model, dropout=dropout)
        self.output_sublayer_seg = SublayerConnection(size=self.d_model, dropout=dropout)
        self.dropout_seg = nn.Dropout(p=dropout)

        self.attention_zone = MultiHeadedAttention(h=self.n_head, d_model=self.d_model, dropout=dropout)
        self.feed_forward_zone = PositionwiseFeedForward(d_model=self.d_model, d_ff=self.d_model * 4, dropout=dropout)
        self.input_sublayer_zone = SublayerConnection(size=self.d_model, dropout=dropout)
        self.output_sublayer_zone = SublayerConnection(size=self.d_model, dropout=dropout)
        self.dropout_zone = nn.Dropout(p=dropout)
        
        self.graph_gru_seg = nn.GRU(self.d_model, gru_hidden_size, num_of_gru_layers, batch_first=True, dropout=dropout)
        self.graph_att_fc1_seg = nn.Linear(in_features=gru_hidden_size, out_features=1)
        self.graph_att_fc2_seg = nn.Linear(in_features=num_of_target_time_feature, out_features=seq_len)
        self.graph_att_bias_seg = nn.Parameter(torch.zeros(1))
        self.graph_att_softmax_seg = nn.Softmax(dim=-1)

        self.graph_gru_zone = nn.GRU(self.d_model, gru_hidden_size, num_of_gru_layers, batch_first=True, dropout=dropout)
        self.graph_att_fc1_zone = nn.Linear(in_features=gru_hidden_size, out_features=1)
        self.graph_att_fc2_zone = nn.Linear(in_features=num_of_target_time_feature, out_features=seq_len)
        self.graph_att_bias_zone = nn.Parameter(torch.zeros(1))
        self.graph_att_softmax_zone = nn.Softmax(dim=-1)

        self.message_share_seg = nn.Sequential(
            nn.Linear(gru_hidden_size, gru_hidden_size),
            nn.ReLU()
        )
        self.message_share_zone = nn.Sequential(
            nn.Linear(gru_hidden_size, gru_hidden_size),
            nn.ReLU()
        )

    def get_timestamp(slef, hour_id):
        timestamp_st = pd.to_datetime("2019-01-01")
        timestamp_cur = timestamp_st + timedelta(hours=hour_id)
        return time.mktime(timestamp_cur.timetuple())

    def get_de_emb(self, tmp_dict, timestamp_his):
        date_str = time.strftime("%Y-%m-%d", time.localtime(int(timestamp_his)))
        date = list(map(float, date_str.split("-")))
        years = torch.tensor([date[0]] * len(tmp_dict)).reshape(-1, 1).cuda()
        months = torch.tensor([date[1]] * len(tmp_dict)).reshape(-1, 1).cuda()
        days = torch.tensor([date[2]] * len(tmp_dict)).reshape(-1, 1).cuda()
        timestamps = torch.tensor([timestamp_his] * len(tmp_dict)).reshape(-1, 1).cuda()

        tmp_idx = torch.tensor(list(tmp_dict.keys())).long().cuda()
        tmp_emb_idx = torch.tensor(list(tmp_dict.values())).cuda()
        
        f_emb = self.de_simple.get_time_embedd(tmp_emb_idx, years, months, days, h_or_t="head", timestamps=timestamps)
        i_emb = self.de_simple.get_time_embedd(tmp_emb_idx, years, months, days, h_or_t="tail", timestamps=timestamps)
        time_emb = (f_emb + i_emb) / 2

        h_emb = self.de_simple.ent_embs_h(tmp_emb_idx)
        t_emb = self.de_simple.ent_embs_t(tmp_emb_idx)
        static_emb = (h_emb + t_emb) / 2

        tmp_emb = torch.zeros(len(tmp_dict), self.node_emb_size).cuda()
        tmp_emb[tmp_idx] = torch.cat([static_emb, time_emb], dim=1)

        return tmp_emb

    def forward(self, graph_feature, road_adj, risk_adj, poi_adj, 
                target_time_feature):
        """
        Arguments:
            graph_feature {Tensor} -- Graph signal matrix，(batch_size, T, D1, N)
            road_adj {np.array} -- segment adjacent matrix，shape：(N, N)
            risk_adj {np.array} -- risk adjacent matrix，shape：(N, N)
            poi_adj {np.array} -- poi adjacent matrix，shape：(N, N)
            target_time_feature {Tensor} -- the feature of target time，shape：(batch_size, num_target_time_feature)
        Returns:
            {Tensor} -- shape：(batch_size, N, gru_hidden_size)
        """        
        batch_size, T, D1, N_seg = graph_feature[0].shape
        _, _, _, N_zone = graph_feature[1].shape
        
        road_graph_output_seg = graph_feature[0].view(-1, D1, N_seg).permute(0, 2, 1).contiguous() # (batch_size * T, N_seg, D1)
        for gcn_layer in self.road_gcn_seg:
            road_graph_output_seg = gcn_layer(road_graph_output_seg, road_adj[0])

        risk_graph_output_seg = graph_feature[0].view(-1, D1, N_seg).permute(0, 2, 1).contiguous()
        for gcn_layer in self.risk_gcn_seg:
            risk_graph_output_seg = gcn_layer(risk_graph_output_seg, risk_adj[0])

        poi_graph_output_seg = graph_feature[0].view(-1, D1, N_seg).permute(0, 2, 1).contiguous()
        for gcn_layer in self.poi_gcn_seg:
            poi_graph_output_seg = gcn_layer(poi_graph_output_seg, poi_adj[0])

        zone_graph_output = graph_feature[1].view(-1, D1, N_zone).permute(0, 2, 1).contiguous() # (batch_size * T, N_zone, D1)
        for gcn_layer in self.zone_gcn:
            zone_graph_output = gcn_layer(zone_graph_output, road_adj[1])

        risk_graph_output_zone = graph_feature[1].view(-1, D1, N_zone).permute(0, 2, 1).contiguous()
        for gcn_layer in self.risk_gcn_zone:
            risk_graph_output_zone = gcn_layer(risk_graph_output_zone, risk_adj[1])

        poi_graph_output_zone = graph_feature[1].view(-1, D1, N_zone).permute(0, 2, 1).contiguous()
        for gcn_layer in self.poi_gcn_zone:
            poi_graph_output_zone = gcn_layer(poi_graph_output_zone, poi_adj[1])

        if self.depth_gcn is True:
            I_seg = torch.eye(self.num_of_graph_node[0]).to(graph_feature[0].device)
            I_zone = torch.eye(self.num_of_graph_node[1]).to(graph_feature[0].device)

            if self.pretrain_kg:
                depth_outputs_zone = None
                dict_hour_zone = {}
                for sample in range(batch_size):
                    hour_id = torch.where(target_time_feature[sample, :24] == 1)[0].item()
                    # print("hour ", hour_id)
                    if hour_id in dict_hour_zone:
                        # print("repeat hour id ", hour_id)
                        depth_zone_adj, depth_poi_adj, depth_risk_adj = dict_hour_zone[hour_id]
                    else:
                        tmp_timestamp = self.get_timestamp(hour_id)

                        zone_emb = self.get_de_emb(self.zone_dict, tmp_timestamp)

                        poi_emb = torch.mm(self.zone_poi_adj.to(graph_feature[0].device), self.get_de_emb(self.poi_dict, tmp_timestamp).to(graph_feature[0].device)) # Y=AX
                        risk_emb = torch.mm(self.zone_rf_adj.to(graph_feature[0].device), self.get_de_emb(self.rf_dict, tmp_timestamp).to(graph_feature[0].device)) # Y=AX

                        depth_zone_adj = F.softmax(F.relu(torch.mm(zone_emb, zone_emb.t())), dim=1) + I_zone
                        depth_poi_adj = F.softmax(F.relu(torch.mm(poi_emb, poi_emb.t())), dim=1) + I_zone
                        depth_risk_adj = F.softmax(F.relu(torch.mm(risk_emb, risk_emb.t())), dim=1) + I_zone

                        dict_hour_zone[hour_id] = (depth_zone_adj, depth_poi_adj, depth_risk_adj)

                    tmp_depth_outputs = None
                    for adj, gcn in zip([depth_zone_adj, depth_risk_adj, depth_poi_adj], [self.depth_zone_gcn, self.depth_risk_gcn_zone, self.depth_poi_gcn_seg]):
                        graph_output = graph_feature[1][sample, :, :, :].view(-1, D1, N_zone).permute(0, 2, 1).contiguous() # (T, N_zone, D1)
                        for gcn_layer in gcn:
                            graph_output = gcn_layer(graph_output, adj)
                        if tmp_depth_outputs is None:
                            tmp_depth_outputs = graph_output.unsqueeze(dim=0)
                        else:
                            tmp_depth_outputs = torch.cat([tmp_depth_outputs, graph_output.unsqueeze(dim=0)], dim=0)
                    # print(tmp_depth_outputs.shape) # (3, T, N_zone, D1)   
                        
                    if depth_outputs_zone is None:
                        depth_outputs_zone = tmp_depth_outputs.unsqueeze(dim=1)
                    else:
                        depth_outputs_zone = torch.cat([depth_outputs_zone, tmp_depth_outputs.unsqueeze(dim=1)], dim=1)
                depth_outputs_zone = depth_outputs_zone.reshape(3, batch_size * T, N_zone, -1)
                
                depth_outputs_seg = None
                dict_hour_seg = {}
                for sample in range(batch_size):
                    hour_id = torch.where(target_time_feature[sample, :24] == 1)[0].item()
                    # print("hour ", hour_id)
                    if hour_id in dict_hour_seg:
                        # print("repeat hour id ", hour_id)
                        depth_seg_adj, depth_poi_adj, depth_risk_adj = dict_hour_seg[hour_id]
                    else:
                        tmp_timestamp = self.get_timestamp(hour_id)

                        seg_emb = self.get_de_emb(self.seg_dict, tmp_timestamp)

                        poi_emb = torch.mm(self.seg_poi_adj.to(graph_feature[0].device), self.get_de_emb(self.poi_dict, tmp_timestamp).to(graph_feature[0].device)) # Y=AX
                        risk_emb = torch.mm(self.seg_rf_adj.to(graph_feature[0].device), self.get_de_emb(self.rf_dict, tmp_timestamp).to(graph_feature[0].device)) # Y=AX

                        depth_seg_adj = F.softmax(F.relu(torch.mm(seg_emb, seg_emb.t())), dim=1) + I_seg
                        depth_poi_adj = F.softmax(F.relu(torch.mm(poi_emb, poi_emb.t())), dim=1) + I_seg
                        depth_risk_adj = F.softmax(F.relu(torch.mm(risk_emb, risk_emb.t())), dim=1) + I_seg

                        dict_hour_seg[hour_id] = (depth_seg_adj, depth_poi_adj, depth_risk_adj)

                    tmp_depth_outputs = None
                    for adj, gcn in zip([depth_seg_adj, depth_risk_adj, depth_poi_adj], [self.depth_seg_gcn, self.depth_risk_gcn_seg, self.depth_poi_gcn_seg]):
                        graph_output = graph_feature[0][sample, :, :, :].view(-1, D1, N_seg).permute(0, 2, 1).contiguous() # (T, N_seg, D1)
                        for gcn_layer in gcn:
                            graph_output = gcn_layer(graph_output, adj)
                        if tmp_depth_outputs is None:
                            tmp_depth_outputs = graph_output.unsqueeze(dim=0)
                        else:
                            tmp_depth_outputs = torch.cat([tmp_depth_outputs, graph_output.unsqueeze(dim=0)], dim=0)
                    # print(tmp_depth_outputs.shape) # (3, T, N_seg, D1)   
                        
                    if depth_outputs_seg is None:
                        depth_outputs_seg = tmp_depth_outputs.unsqueeze(dim=1)
                    else:
                        depth_outputs_seg = torch.cat([depth_outputs_seg, tmp_depth_outputs.unsqueeze(dim=1)], dim=1)
                depth_outputs_seg = depth_outputs_seg.reshape(3, batch_size * T, N_seg, -1)
            else:
                depth_seg_adj = F.softmax(F.relu(torch.mm(self.seg_emb.weight, self.seg_emb.weight.t())), dim=1) + I_seg
                depth_poi_adj_seg = F.softmax(F.relu(torch.mm(self.poi_emb_seg.weight, self.poi_emb_seg.weight.t())), dim=1) + I_seg
                depth_risk_adj_seg = F.softmax(F.relu(torch.mm(self.risk_emb_seg.weight, self.risk_emb_seg.weight.t())), dim=1) + I_seg

                depth_outputs_seg = []
                for adj, gcn in zip([depth_seg_adj, depth_risk_adj_seg, depth_poi_adj_seg], [self.depth_seg_gcn, self.depth_risk_gcn_seg, self.depth_poi_gcn_seg]):
                    graph_output = graph_feature[0].view(-1, D1, N_seg).permute(0, 2, 1).contiguous()
                    for gcn_layer in gcn:
                        graph_output = gcn_layer(graph_output, adj)
                    depth_outputs_seg.append(graph_output)

                depth_zone_adj = F.softmax(F.relu(torch.mm(self.zone_emb.weight, self.zone_emb.weight.t())), dim=1) + I_zone
                depth_poi_adj_zone = F.softmax(F.relu(torch.mm(self.poi_emb_zone.weight, self.poi_emb_zone.weight.t())), dim=1) + I_zone
                depth_risk_adj_zone = F.softmax(F.relu(torch.mm(self.risk_emb_zone.weight, self.risk_emb_zone.weight.t())), dim=1) + I_zone

                depth_outputs_zone = []
                for adj, gcn in zip([depth_zone_adj, depth_risk_adj_zone, depth_poi_adj_zone], [self.depth_seg_gcn, self.depth_risk_gcn_zone, self.depth_poi_gcn_zone]):
                    graph_output = graph_feature[1].view(-1, D1, N_zone).permute(0, 2, 1).contiguous()
                    for gcn_layer in gcn:
                        graph_output = gcn_layer(graph_output, adj)
                    depth_outputs_zone.append(graph_output)
            
            shallow_outputs_seg = [road_graph_output_seg, risk_graph_output_seg, poi_graph_output_seg]
            shallow_outputs_zone = [zone_graph_output, risk_graph_output_zone, poi_graph_output_zone]

            fusion_outputs_seg = []
            for sh, de, w in zip(shallow_outputs_seg, depth_outputs_seg, self.gcn_weights_seg):
                if self.prefer_depth:
                    fusion_outputs_seg.append(w * sh + de)
                else:
                    fusion_outputs_seg.append(sh + w * de)
            graph_output_seg = torch.stack(fusion_outputs_seg, dim=3).reshape(batch_size * T, N_seg, -1)

            fusion_outputs_zone = []
            for sh, de, w in zip(shallow_outputs_zone, depth_outputs_zone, self.gcn_weights_zone):
                if self.prefer_depth:
                    fusion_outputs_zone.append(w * sh + de)
                else:
                    fusion_outputs_zone.append(sh + w * de)
                # fusion_outputs.append((1 - w) * sh + w * de)
            graph_output_zone = torch.stack(fusion_outputs_zone, dim=3).reshape(batch_size * T, N_zone, -1)
        else:
            graph_output_seg = torch.stack([road_graph_output_seg, risk_graph_output_seg, poi_graph_output_seg], dim=3).reshape(batch_size * T, N_seg, -1)
            graph_output_zone = torch.stack([zone_graph_output, risk_graph_output_zone, poi_graph_output_zone], dim=3).reshape(batch_size * T, N_zone, -1)
        

        x = graph_output_zone # (batch_size * T, seq_len=N_zone, d_model)
        for _ in range(self.encoder_layer): # 迭代计算多轮，参数共享，反向传播梯度时，根据链式法则，（L(x_t, label)->x_t->x_{t-1}->...->x_3->x_2->x_1->w）逐轮求导即可！？
            x = self.input_sublayer_zone(x, lambda _x: self.attention_zone.forward(_x, _x, _x))
            x = self.dropout_zone(self.output_sublayer_zone(x, self.feed_forward_zone))
        graph_output_zone = x

        x = graph_output_seg # (batch_size * T, seq_len=N_seg, d_model)
        for _ in range(self.encoder_layer):
            x = self.input_sublayer_seg(x, lambda _x: self.attention_seg.forward(_x, _x, _x))
            x = self.dropout_seg(self.output_sublayer_seg(x, self.feed_forward_seg))
        graph_output_seg = x

        graph_output_seg = graph_output_seg.view(batch_size, T, N_seg, -1)\
                                    .permute(0, 2, 1, 3)\
                                    .contiguous()\
                                    .view(batch_size * N_seg, T, -1)
        graph_output_seg, _ = self.graph_gru_seg(graph_output_seg)
        graph_output_zone = graph_output_zone.view(batch_size, T, N_zone, -1)\
                                    .permute(0, 2, 1, 3)\
                                    .contiguous()\
                                    .view(batch_size * N_zone, T, -1)
        graph_output_zone, _ = self.graph_gru_zone(graph_output_zone)

        graph_target_time = torch.unsqueeze(target_time_feature, 1).repeat(1, N_seg, 1).view(batch_size * N_seg, -1)
        graph_att_fc1_output_seg = torch.squeeze(self.graph_att_fc1_seg(graph_output_seg))
        graph_att_fc2_output_seg = self.graph_att_fc2_seg(graph_target_time)
        graph_att_score_seg = self.graph_att_softmax_seg(F.relu(graph_att_fc1_output_seg + graph_att_fc2_output_seg + self.graph_att_bias_seg))
        graph_att_score_seg = graph_att_score_seg.view(batch_size * N_seg, -1, 1)
        graph_output_seg = torch.sum(graph_output_seg * graph_att_score_seg, dim=1)
        graph_output_seg = graph_output_seg.view(batch_size, N_seg, -1).contiguous()

        graph_target_time = torch.unsqueeze(target_time_feature, 1).repeat(1, N_zone, 1).view(batch_size * N_zone, -1)
        graph_att_fc1_output_zone = torch.squeeze(self.graph_att_fc1_zone(graph_output_zone))
        graph_att_fc2_output_zone = self.graph_att_fc2_zone(graph_target_time)
        graph_att_score_zone = self.graph_att_softmax_zone(F.relu(graph_att_fc1_output_zone + graph_att_fc2_output_zone + self.graph_att_bias_zone))
        graph_att_score_zone = graph_att_score_zone.view(batch_size * N_zone, -1, 1)
        graph_output_zone = torch.sum(graph_output_zone * graph_att_score_zone, dim=1)
        graph_output_zone = graph_output_zone.view(batch_size, N_zone, -1).contiguous()

        graph_output_seg = graph_output_seg + self.message_share_seg(torch.matmul(self.zone2seg.permute(0, 2, 1).to(graph_output_zone.device), graph_output_zone))
        graph_output_zone = graph_output_zone + self.message_share_zone(torch.matmul(self.zone2seg.to(graph_output_seg.device), graph_output_seg))

        return graph_output_seg, graph_output_zone

class STKGRisk(nn.Module):
    def __init__(self, num_of_gru_layers, seq_len, pre_len, 
                gru_hidden_size, num_of_target_time_feature, 
                num_of_graph_feature, share_feature, nums_of_graph_filters,
                num_of_graph_node, squeeze_dim, depth_gcn=False, inverse=False, scaler=None,loss_weight=0.5,complicate=False, hyper_hidden_size=64,
                use_sigmoid=False, prefer_depth=False, transform=False,decoder="ori", component=64, conv_hidden_size=64,
                pretrain_kg=False, poi_feature=None, use_all_fea=False, encoder_layer=3, dropout=0.1, n_head=4,zone2seg=None,
                depth_adjs_path=None, kg_model_path=None, num_experts=64, ):
        """
        Arguments:
            num_of_gru_layers {int} -- the number of GRU layers
            seq_len {int} -- the time length of input
            pre_len {int} -- the time length of prediction
            gru_hidden_size {int} -- the hidden size of GRU
            num_of_target_time_feature {int} -- the number of target time feature，为 24(hour) + 7(week) + 1(holiday) = 32
            num_of_graph_feature {int} -- the number of graph node feature，(batch_size, seq_len, D, N), num_of_graph_feature=D
            nums_of_graph_filters {list} -- the number of GCN output feature
        """
        super(STKGRisk, self).__init__()
        self.use_all_fea = use_all_fea
        self.poi_feature = poi_feature
        self.num_of_graph_node = num_of_graph_node
        self.transform = transform
        self.zone2seg = zone2seg

        self.decoder = decoder
        self.inverse = inverse
        self.scaler = scaler

        if self.transform:
            self.trans_seg = nn.Sequential(
                nn.Linear(num_of_graph_feature, 64),
                nn.ReLU(),
                nn.Linear(64, num_of_graph_feature)
            )
            self.trans_zone = nn.Sequential(
                nn.Linear(num_of_graph_feature, 64),
                nn.ReLU(),
                nn.Linear(64, num_of_graph_feature)
            )

        self.mmoe = MMOE(input_size=share_feature, num_experts=num_experts, experts_out=gru_hidden_size, experts_hidden=gru_hidden_size * 4)

        self.STGN = STGN(num_of_graph_feature, nums_of_graph_filters, 
                                        seq_len, num_of_gru_layers, gru_hidden_size, 
                                        num_of_target_time_feature, num_of_graph_node, 
                                        depth_gcn=depth_gcn, prefer_depth=prefer_depth, zone2seg=zone2seg,
                                        pretrain_kg=pretrain_kg, encoder_layer=encoder_layer, dropout=dropout, n_head=n_head,
                                        depth_adjs_path=depth_adjs_path, kg_model_path=kg_model_path)
        
        if self.decoder == "ori":
            self.graph_weight_seg = nn.Linear(gru_hidden_size, squeeze_dim)
            self.graph_weight_zone = nn.Linear(gru_hidden_size, squeeze_dim)
            self.output_layer_seg = nn.Linear(squeeze_dim * num_of_graph_node[0], pre_len * num_of_graph_node[0])
            self.output_layer_zone = nn.Linear(squeeze_dim * num_of_graph_node[1], pre_len * num_of_graph_node[1])
        
        self.use_sigmoid = use_sigmoid 
        if self.use_sigmoid:
            self.sigmoid = nn.Sigmoid()
        
        

        self.fcn_seg = nn.Sequential(
                    nn.Linear(num_of_graph_node[0], num_of_graph_node[0] // 4),
                    nn.ReLU(),
                    nn.Linear(num_of_graph_node[0] // 4, num_of_graph_node[0])
                )
        self.fcn_zone = nn.Sequential(
                    nn.Linear(num_of_graph_node[1], num_of_graph_node[1] // 4),
                    nn.ReLU(),
                    nn.Linear(num_of_graph_node[1] // 4, num_of_graph_node[1])
                )

        if self.decoder == "zip":
            self.loss_weight = loss_weight
            self.component = component
            if complicate:
                self.hypernet = Hypernet(gru_hidden_size, hidden_sizes=[hyper_hidden_size, hyper_hidden_size], param_sizes=[self.component, self.component])
                self.get_pai = nn.Sequential(
                                nn.Conv2d(in_channels=gru_hidden_size,
                                    out_channels=conv_hidden_size,
                                    kernel_size=(1,1),
                                    bias=True),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=conv_hidden_size,
                                    out_channels=conv_hidden_size,
                                    kernel_size=(1,1),
                                    bias=True),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=conv_hidden_size,
                                    out_channels=pre_len,
                                    kernel_size=(1,1),
                                    bias=True),
                                nn.ReLU()
                            )
            else:
                self.hypernet_seg = Hypernet(gru_hidden_size, param_sizes=[self.component, self.component])
                self.hypernet_zone = Hypernet(gru_hidden_size, param_sizes=[self.component, self.component])
                self.get_pai = nn.Conv2d(in_channels=gru_hidden_size,
                                    out_channels=pre_len,
                                    kernel_size=(1,1),
                                    bias=True)
    
    def compute_loss(self, predicts, labels, top_mask, trans=False):
        """
        
        Arguments:
            predicts {Tensor} -- predict，(batch_size, pre_len, N)
            labels {Tensor} -- label，(batch_size, pre_len, N)
            top_mask {np.array} -- mask matrix，(N)
        
        Returns:
            {Tensor} -- MSELoss, (1, )
        """
        is_zone = False
        if top_mask.shape[0] < 128:
            is_zone = True

        loss = (labels - predicts) ** 2

        ratio_mask = torch.zeros(labels.shape).to(predicts.device)
        if trans is False:
            index_1 = labels == 0
            index_2 = (labels > 0) & (labels <= 0.125) # 0 ~ 1
            index_3 = (labels > 0.125) & (labels <= 0.25) # 1 ~ 2
            index_4 = (labels > 0.25) & (labels <= 0.375) # 2 ~ 3
            index_5  = (labels > 0.375) # > 3
        else:
            index_1 = labels == 0
            index_2 = (labels > 0) & (labels <= 1) # 0 ~ 1
            index_3 = (labels > 1) & (labels <= 2) # 1 ~ 2
            index_4 = (labels > 2) & (labels <= 3) # 2 ~ 3
            index_5  = (labels > 3) # > 3

        if is_zone:
            ratio_mask[index_1] = 1
            ratio_mask[index_2] = 5
            ratio_mask[index_3] = 10
            ratio_mask[index_4] = 15
            ratio_mask[index_5] = 5
        else:
            ratio_mask[index_1] = 1
            ratio_mask[index_2] = 20
            ratio_mask[index_3] = 30
            ratio_mask[index_4] = 50
            ratio_mask[index_5] = 100

        top_mask = torch.from_numpy(top_mask).to(predicts.device)
        ratio_mask = torch.where(top_mask == 0, torch.tensor(0.5).to(predicts.device), ratio_mask)

        loss *= ratio_mask

        return torch.mean(loss)

    def forward(self, target_time_feature, graph_feature, x_factor, x_time, x_weather,
                road_adj, risk_adj, poi_adj, train_label=None, risk_mask=None):
        """
        Arguments:
            graph_feature {Tensor} -- Graph signal matrix，(batch_size, T, D1, N)
            target_time_feature {Tensor} -- the feature of target time，shape：(batch_size, num_target_time_feature)
            x_factor, x_time, x_weather {Tensor} -- the external features (batch_size, T, D')
            road_adj {np.array} -- segment adjacent matrix，shape：(N, N)
            risk_adj {np.array} -- risk adjacent matrix，shape：(N, N)
            poi_adj {np.array} -- poi adjacent matrix，shape：(N, N)
        Returns:
            {Tensor} -- shape：(batch_size, pre_len, N)
        """
        batch_size, T, _, N_seg  = graph_feature[0].shape
        _, _, _, N_zone = graph_feature[1].shape

        if self.use_all_fea:
            poi_fea_seg = self.poi_feature[0].repeat(batch_size, T, 1, 1).to(target_time_feature.device)
            poi_fea_zone = self.poi_feature[1].repeat(batch_size, T, 1, 1).to(target_time_feature.device)

            graph_feature_seg = torch.cat([graph_feature[0], poi_fea_seg], dim=2).to(torch.float)
            graph_feature_zone = torch.cat([graph_feature[1], poi_fea_zone], dim=2).to(torch.float)

        if self.transform:
            graph_feature_seg = self.trans_seg(graph_feature_seg.permute(0, 1, 3, 2)).permute(0, 1, 3, 2) 
            graph_feature_zone = self.trans_zone(graph_feature_zone.permute(0, 1, 3, 2)).permute(0, 1, 3, 2) 

        factor_fea = x_factor.to(target_time_feature.device)
        time_fea = x_time.to(target_time_feature.device)
        weather_fea = x_weather.to(target_time_feature.device)
        global_x = torch.cat([factor_fea, time_fea, weather_fea], dim = 2)

        global_info = self.mmoe(global_x.reshape(batch_size * T, -1)).reshape(batch_size, T, -1)
        global_info = global_info.sum(dim=1, keepdims=True)
        # print(global_info.shape) # torch.Size([batch_size, recent_prior + week_prior, gru_hidden_size])

        graph_output_seg, graph_output_zone = self.STGN([graph_feature_seg, graph_feature_zone], road_adj, risk_adj, poi_adj, target_time_feature)
        graph_output_seg = graph_output_seg + global_info
        graph_output_zone = graph_output_zone + global_info
        # print(graph_output_seg.shape, graph_output_zone.shape)

        if self.decoder == "ori":
            graph_output_seg = self.graph_weight_seg(graph_output_seg).view(batch_size, -1)
            final_output_seg = self.output_layer_seg(graph_output_seg)\
                                .view(batch_size, -1, self.num_of_graph_node[0])
            graph_output_zone = self.graph_weight_zone(graph_output_zone).view(batch_size, -1)
            final_output_zone = self.output_layer_zone(graph_output_zone)\
                                .view(batch_size, -1, self.num_of_graph_node[1])

            consistent_loss = self.compute_loss(torch.matmul(final_output_seg, self.zone2seg.to(final_output_seg.device).t()), final_output_zone, risk_mask[1])

            if self.use_sigmoid:
                seg_loss = self.compute_loss(self.sigmoid(final_output_seg), train_label[0], risk_mask[0])
                zone_loss = self.compute_loss(self.sigmoid(final_output_zone), train_label[1], risk_mask[1])
                all_loss = seg_loss + zone_loss + consistent_loss
                return all_loss, self.sigmoid(final_output_seg), self.sigmoid(final_output_zone)
            else:
                seg_loss = self.compute_loss(final_output_seg, train_label[0], risk_mask[0])
                zone_loss = self.compute_loss(final_output_zone, train_label[1], risk_mask[1])
                all_loss = seg_loss + zone_loss + consistent_loss
                return all_loss, final_output_seg, final_output_zone
        elif self.decoder == "zip":
            weights_seg, lamb_seg = self.hypernet_seg(graph_output_seg)
            pi_seg = self.get_pai(graph_output_seg.permute(0, 2, 1).unsqueeze(dim=-1)).squeeze(dim=-1) # (batch_size, pre_len, N_seg)
            pi_seg = F.sigmoid(pi_seg)
            lamb_seg = F.softplus(lamb_seg) # (batch_size, N_seg, component) 
            weights_seg = F.softmax(weights_seg, dim=-1)
            final_output_seg = (1 - pi_seg) * ((torch.sum(weights_seg * lamb_seg, dim=-1)).unsqueeze(dim=1)) # (batch_size, pre_len, N_seg)

            weights_zone, lamb_zone = self.hypernet_zone(graph_output_zone)
            pi_zone = self.get_pai(graph_output_zone.permute(0, 2, 1).unsqueeze(dim=-1)).squeeze(dim=-1) # (batch_size, pre_len, N_zone)
            pi_zone = F.sigmoid(pi_zone)
            lamb_zone = F.softplus(lamb_zone) # (batch_size, N_zone, component)
            weights_zone = F.softmax(weights_zone, dim=-1)
            final_output_zone = (1 - pi_zone) * ((torch.sum(weights_zone * lamb_zone, dim=-1)).unsqueeze(dim=1)) # (batch_size, pre_len, N_zone)
            # final_output_zone = self.fcn_zone(final_output_zone)

            consistent_loss = self.compute_loss(torch.matmul(final_output_seg, self.zone2seg.to(final_output_seg.device).t()), final_output_zone, risk_mask[1])

            if self.use_sigmoid:
                nll_seg = mix_zip_nll_loss(weights_seg, train_label[0], pi_seg, lamb_seg, y_mask=risk_mask[0])
                nll_zone = mix_zip_nll_loss(weights_zone, train_label[1], pi_zone, lamb_zone, y_mask=risk_mask[1])
                mse_seg = self.compute_loss(self.sigmoid(final_output_seg), train_label[0], risk_mask[0])
                mse_zone = self.compute_loss(self.sigmoid(final_output_zone), train_label[1], risk_mask[1])

                seg_loss = (1 - self.loss_weight) * nll_seg + self.loss_weight * mse_seg
                zone_loss = (1 - self.loss_weight) * nll_zone + self.loss_weight * mse_zone
                all_loss = consistent_loss + seg_loss + zone_loss

                return all_loss, self.sigmoid(final_output_seg), self.sigmoid(final_output_zone)
            else:
                if self.inverse:
                    train_label[0] = self.scaler[0].inverse_transform(train_label[0])
                    train_label[1] = self.scaler[1].inverse_transform(train_label[1]) 
                
                nll_seg = mix_zip_nll_loss(weights_seg, train_label[0], pi_seg, lamb_seg, y_mask=risk_mask[0])
                nll_zone = mix_zip_nll_loss(weights_zone, train_label[1], pi_zone, lamb_zone, y_mask=risk_mask[1])

                mse_seg = self.compute_loss(final_output_seg, train_label[0], risk_mask[0], trans=self.inverse)
                mse_zone = self.compute_loss(final_output_zone, train_label[1], risk_mask[1], trans=self.inverse) 

                seg_loss = (1 - self.loss_weight) * nll_seg + self.loss_weight * mse_seg
                zone_loss = (1 - self.loss_weight) * nll_zone + self.loss_weight * mse_zone
                all_loss = consistent_loss + seg_loss + zone_loss

                return all_loss, final_output_seg, final_output_zone
