import numpy as np
import pandas as pd
import torch
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from lib.metrics import mask_evaluation_np

def mix_zip_nll_loss(weights, train_label, pi, lamb, y_mask=None, scale=1, eps=1e-10):
    """
    y: true values
    y_mask: whether missing mask is given
    """

    y_mask = torch.from_numpy(y_mask).to(pi.device)
    y = torch.where(y_mask == 0, torch.tensor(-1.0).to(pi.device), train_label)

    y = y.reshape(-1) # batch_size * pre_len=1 * node_num
    
    batch_size, node_num, component = weights.shape
    weights = weights.reshape(batch_size * node_num, component)
    pi = pi.reshape(batch_size * node_num)
    lamb = lamb.reshape(batch_size * node_num, component)

    idx_yeq0 = y == 0
    idx_yg0  = y > 0
    
    lamb_yeq0 = lamb[idx_yeq0, :]
    pi_yeq0 = pi[idx_yeq0]
    w_yeq0 = weights[idx_yeq0, :]

    lamb_yg0 = lamb[idx_yg0, :]
    pi_yg0 = pi[idx_yg0]
    yg0 = y[idx_yg0]
    w_g0 = weights[idx_yg0, :]

    nll = torch.tensor(0).float().to(y.device)
    if pi_yeq0.shape[0] > 0:
        L_yeq0 = torch.log(pi_yeq0 + (1 - pi_yeq0) * (w_yeq0 * torch.exp(-lamb_yeq0)).sum(dim=-1) + eps)

        nll += -torch.sum(L_yeq0)

    if pi_yg0.shape[0] > 0:
        yg0 = yg0.unsqueeze(dim=-1)
        L_yg0  = torch.log(1 - pi_yg0 + eps) + ((torch.log(w_g0 + eps) + yg0 * torch.log(lamb_yg0 + eps) - torch.lgamma(yg0 + 1) - lamb_yg0).exp().sum(dim=-1) + eps).log()
        nll += -torch.sum(L_yg0) * scale

    return nll

def min_max_transform(data):
    tmp = np.transpose(data, (0, 2, 1)).reshape((-1, data.shape[1]))
    max_ = np.max(tmp, axis=0)
    min_ = np.min(tmp, axis=0)

    T, D, N = data.shape
    data = np.transpose(data, (0, 2, 1)).reshape((-1, D))
    data = (data - min_) / (max_ - min_)

    return np.transpose(data.reshape((T, N, -1)), (0, 2, 1))

class Scaler_Bro_and_Man:
    def __init__(self,  train):
        """ BROOKLYN/MANHATTON Max - Min
        
        Arguments:
            train {np.ndarray} -- shape(T,  D,  N)
        """
        train_temp = np.transpose(train, (0, 2, 1)).reshape((-1, train.shape[1]))
        self.max = np.max(train_temp, axis=0)
        self.min = np.min(train_temp, axis=0)
    def transform(self,  data):
        """norm train，valid，test
        
        Arguments:
            data {np.ndarray} --  shape(T,  D,  N)
        
        Returns:
            {np.ndarray} -- shape(T,  D,  N)
        """
        T, D, N = data.shape
        data = np.transpose(data, (0, 2, 1)).reshape((-1, D))
        data[:, 0] = (data[:, 0] - self.min[0]) / (self.max[0] - self.min[0])
        data[:, 1] = (data[:, 1] - self.min[1]) / (self.max[1] - self.min[1])
        return np.transpose(data.reshape((T, N, -1)), (0, 2, 1))
    
    def inverse_transform(self, data):
        """
        Arguments:
            data {np.ndarray} --  shape(T,  D,  N)
        
        Returns:
            {np.ndarray} --  shape (T,  D,  N)
        """
        return data * (self.max[0] - self.min[0]) + self.min[0]

def clamp_preserve_gradients(x: torch.Tensor, min: float, max: float) -> torch.Tensor:
    """Clamp the tensor while preserving gradients in the clamped region."""
    return x + (x.clamp(min, max) - x).detach()

@torch.no_grad()
def compute_loss(net, dataloader, risk_mask, road_adj, risk_adj, poi_adj, 
                global_step, device, num_of_graph_node, inverse, scaler):
    """compute val/test loss
    
    Arguments:
        net {Molde} -- model
        dataloader {DataLoader} -- val/test dataloader
        risk_mask {np.array} -- mask matrix，shape(W, H)
        road_adj  {np.array} -- road adjacent matrix，shape(N, N)
        risk_adj  {np.array} -- risk adjacent matrix，shape(N, N)
        poi_adj  {np.array} -- poi adjacent matrix，shape(N, N)
        global_step {int} -- global_step
        device {Device} -- GPU
    
    Returns:
        np.float32 -- mean loss
    """
    net.eval()
    temp = []
    seg_preds = []
    seg_labels = []
    zone_preds = []
    zone_labels = []
    for target_time, graph_feature, label, x_factor, x_time, x_weather in dataloader:
        target_time, graph_feature, label = target_time.to(device), graph_feature.to(device), label.to(device)
        x_factor, x_time, x_weather = x_factor.to(device), x_time.to(device), x_weather.to(device)

        seg_graph_feature, seg_label = graph_feature[:, :, :, :num_of_graph_node[0]], label[:, :, :num_of_graph_node[0]]
        zone_graph_feature, zone_label = graph_feature[:, :, :, num_of_graph_node[0]:], label[:, :, num_of_graph_node[0]:]
        
        l, pred_seg, pred_zone = net(target_time, [seg_graph_feature, zone_graph_feature], x_factor, x_time, x_weather, road_adj, risk_adj, poi_adj, [seg_label, zone_label], risk_mask)
        # print("val loss:", l.shape, pred.shape, label.shape)
        
        seg_preds.append(pred_seg.detach().cpu().numpy())
        seg_labels.append(seg_label.detach().cpu().numpy())
        zone_preds.append(pred_zone.detach().cpu().numpy())
        zone_labels.append(zone_label.detach().cpu().numpy())
        
        temp.append(l.mean().cpu().item())
    loss_mean = sum(temp) / len(temp)

    seg_preds = np.concatenate(seg_preds,  0)
    seg_labels = np.concatenate(seg_labels,  0)
    zone_preds = np.concatenate(zone_preds,  0)
    zone_labels = np.concatenate(zone_labels,  0)

    if inverse:
        seg_preds = seg_preds
        zone_preds = zone_preds
    else:
        seg_preds = scaler[0].inverse_transform(seg_preds)
        zone_preds = scaler[1].inverse_transform(zone_preds)
    seg_labels = scaler[0].inverse_transform(seg_labels)
    zone_labels = scaler[1].inverse_transform(zone_labels)

    val_rmse_seg, val_recall_seg, val_map_seg = mask_evaluation_np(seg_labels, seg_preds, risk_mask[0], null_val=0, test=False, is_zone=False)
    val_rmse_zone, val_recall_zone, val_map_zone = mask_evaluation_np(zone_labels, zone_preds, risk_mask[1], null_val=0, test=False, is_zone=True)

    print(f"val {global_step} step, Segment rmse {val_rmse_seg}, recall {val_recall_seg}, map {val_map_seg}, loss_mean {loss_mean}", flush=True)
    print(f"val {global_step} step, Zone rmse {val_rmse_zone}, recall {val_recall_zone}, map {val_map_zone}, loss_mean {loss_mean}", flush=True)

    seg_val = 1000 * val_rmse_seg - val_recall_seg * 10 - val_map_seg * 3
    zone_val = 100 * val_rmse_zone - val_recall_zone * 8 - val_map_zone * 2

    return  seg_val + zone_val

@torch.no_grad()
def predict_and_evaluate(net, dataloader, risk_mask, road_adj, risk_adj, poi_adj, 
                        scaler, device, num_of_graph_node, inverse=True):
    """predict val/test,  return metrics
    
    Arguments:
        net {Model} -- model
        dataloader {DataLoader} -- val/test dataloader
        risk_mask {np.array} -- mask matrix，shape(N)
        road_adj  {np.array} -- road adjacent matrix，shape(N, N)
        risk_adj  {np.array} -- risk adjacent matrix，shape(N, N)
        poi_adj  {np.array} -- poi adjacent matrix，shape(N, N)
        global_step {int} -- global_step
        scaler {Scaler} -- record max and min
        device {Device} -- GPU
    
    Returns:
        np.float32 -- RMSE，Recall，MAP
        np.array -- label and pre，shape(num_sample, pre_len, N)

    """
    net.eval()
    seg_preds = []
    seg_labels = []
    zone_preds = []
    zone_labels = []
    for target_time, graph_feature, label, x_factor, x_time, x_weather in dataloader:
        target_time, graph_feature, label = target_time.to(device), graph_feature.to(device), label.to(device)
        x_factor, x_time, x_weather = x_factor.to(device), x_time.to(device), x_weather.to(device)

        seg_graph_feature, seg_label = graph_feature[:, :, :, :num_of_graph_node[0]], label[:, :, :num_of_graph_node[0]]
        zone_graph_feature, zone_label = graph_feature[:, :, :, num_of_graph_node[0]:], label[:, :, num_of_graph_node[0]:]

        _, pred_seg, pred_zone = net(target_time, [seg_graph_feature, zone_graph_feature], x_factor, x_time, x_weather, road_adj, risk_adj, poi_adj, [seg_label, zone_label], risk_mask)
        
        seg_preds.append(pred_seg.detach().cpu().numpy())
        seg_labels.append(seg_label.detach().cpu().numpy())
        zone_preds.append(pred_zone.detach().cpu().numpy())
        zone_labels.append(zone_label.detach().cpu().numpy())
    seg_preds = np.concatenate(seg_preds,  0)
    seg_labels = np.concatenate(seg_labels,  0)
    zone_preds = np.concatenate(zone_preds,  0)
    zone_labels = np.concatenate(zone_labels,  0)

    if inverse:
        inverse_trans_pre_seg = seg_preds
        inverse_trans_pre_zone = zone_preds
    else:
        inverse_trans_pre_seg = scaler[0].inverse_transform(seg_preds)
        inverse_trans_pre_zone = scaler[1].inverse_transform(zone_preds)
    inverse_trans_label_seg = scaler[0].inverse_transform(seg_labels)
    inverse_trans_label_zone = scaler[1].inverse_transform(zone_labels)

    rmse_zone, recall_zone, map_zone = mask_evaluation_np(inverse_trans_label_zone, inverse_trans_pre_zone, risk_mask[1], null_val=0, is_zone=True)
    rmse_seg, recall_seg, map_seg = mask_evaluation_np(inverse_trans_label_seg, inverse_trans_pre_seg, risk_mask[0], null_val=0, is_zone=False)
    
    return [rmse_seg, rmse_zone], [recall_seg, recall_zone], [map_seg, map_zone], [inverse_trans_pre_seg, inverse_trans_pre_zone], [inverse_trans_label_seg, inverse_trans_label_zone]