import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import json
import configparser
import pickle as pkl
from time import time
from datetime import datetime
import shutil
import argparse
import random
import math
from collections import Counter
import sys
import os
import nni

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
print(curPath, rootPath)
# sys.path.append(rootPath)
sys.path.append(curPath)

from lib.dataloader import generate_dataset
from lib.early_stop import EarlyStopping
from model.STKGRisk import STKGRisk
from lib.utils import compute_loss, predict_and_evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help='configuration file')
parser.add_argument("--gpus", type=str, help="test program", default="0")
parser.add_argument("--test", action="store_true", help="test mode")
parser.add_argument("--use_sigmoid", action="store_true")
parser.add_argument("--depth_gcn", action="store_true")
parser.add_argument("--pretrain_kg", action="store_true")
parser.add_argument("--use_nni", action="store_true")
parser.add_argument("--aug", action="store_true")
parser.add_argument("--k_neighbors", type=int, default=3)
parser.add_argument("--check_result", action="store_true")
parser.add_argument("--inverse", action="store_true")
parser.add_argument("--complicate", action="store_true")
parser.add_argument("--no_top_mask", action="store_true") # no top mask  
parser.add_argument("--part_top_mask", action="store_true") # part top mask  
parser.add_argument("--clip_grad", action="store_true") # clip grad

args = parser.parse_args()
config_filename = args.config
with open(config_filename, 'r') as f:
    config = json.loads(f.read())
print(json.dumps(config, indent=4))

if args.use_nni:
    import nni
    params = nni.get_next_parameter()
    nni_dir = "/mnt/nfs-storage/tkg_and_risk/"

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

all_data_filename_zone = config['all_data_filename_zone']
mask_filename_zone = config['mask_filename_zone']
all_data_filename_seg = config['all_data_filename_seg']
mask_filename_seg = config['mask_filename_seg']
adj_filename_zone = config['adj_filename_zone']
adj_filename_seg = config['adj_filename_seg']
zone2segment_filename = config['zone2segment_filename']

local_dir = "../"
if args.use_nni:
    all_data_filename_seg = os.path.join(nni_dir, all_data_filename_seg)
    mask_filename_seg = os.path.join(nni_dir, mask_filename_seg)
    adj_filename_seg = os.path.join(nni_dir, adj_filename_seg)
    all_data_filename_zone = os.path.join(nni_dir, all_data_filename_zone)
    mask_filename_zone = os.path.join(nni_dir, mask_filename_zone)
    adj_filename_zone = os.path.join(nni_dir, adj_filename_zone)
    zone2segment_filename = os.path.join(nni_dir, zone2segment_filename)
else:
    all_data_filename_seg = os.path.join(local_dir, all_data_filename_seg)
    mask_filename_seg = os.path.join(local_dir, mask_filename_seg)
    adj_filename_seg = os.path.join(local_dir, adj_filename_seg)
    all_data_filename_zone = os.path.join(local_dir, all_data_filename_zone)
    mask_filename_zone = os.path.join(local_dir, mask_filename_zone)
    adj_filename_zone = os.path.join(local_dir, adj_filename_zone)
    zone2segment_filename = os.path.join(local_dir, zone2segment_filename)

if args.aug:
    print(f"augment data with {args.k_neighbors}")

    print("before aug directory", all_data_filename_seg)
    all_data_filename_seg = all_data_filename_seg[:-4] + "_aug" + all_data_filename_seg[-4:]
    all_data_filename_zone = all_data_filename_zone[:-4] + "_aug" + all_data_filename_zone[-4:]
    print("after aug directory", all_data_filename_seg)

    adj_filename_seg = adj_filename_seg[:-4] + "_aug" + adj_filename_seg[-4:]
    adj_filename_zone = adj_filename_zone[:-4] + "_aug" + adj_filename_zone[-4:]

patience = config['patience']
delta = config['delta']

if config['seed'] is not None:
    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.backends.cudnn.deterministic=True
    np.random.seed(seed)
    random.seed(seed)

train_rate = config['train_rate']
valid_rate = config['valid_rate']

recent_prior = config['recent_prior']
week_prior = config['week_prior']
one_day_period = config['one_day_period']
days_of_week = config['days_of_week']
pre_len = config['pre_len']
seq_len = recent_prior + week_prior

training_epoch = config['training_epoch']

def training(net,
            training_epoch,
            train_loader,
            val_loader,
            test_loader,
            high_test_loader,
            road_adj,
            risk_adj,
            poi_adj,
            risk_mask,
            trainer,
            early_stop,
            device,
            scaler,
            num_of_graph_node,
            data_type="manhatton",
            ):
    global_step = 1
    is_nan = False

    for epoch in range(1, training_epoch + 1):
        net.train()

        if is_nan:
            break

        batch = 1

        for target_time, graph_feature, train_label, x_factor, x_time, x_weather in train_loader:
            start_time = time()
            target_time, graph_feature, train_label = target_time.to(device), graph_feature.to(device), train_label.to(device)
            x_factor, x_time, x_weather = x_factor.to(device), x_time.to(device), x_weather.to(device)

            seg_graph_feature, seg_train_label = graph_feature[:, :, :, :num_of_graph_node[0]], train_label[:, :, :num_of_graph_node[0]]
            zone_graph_feature, zone_train_label = graph_feature[:, :, :, num_of_graph_node[0]:], train_label[:, :, num_of_graph_node[0]:]

            l = net(target_time, [seg_graph_feature, zone_graph_feature], x_factor, x_time, x_weather, road_adj, risk_adj, poi_adj, [seg_train_label, zone_train_label], risk_mask)[0]
            # print("train loss:", l.shape)
            
            if torch.isnan(l).sum() > 0:
                print("NAN Value")
                is_nan = True
                break
            
            trainer.zero_grad()
            
            l.mean().backward() # 多次求均值结果一样！
            if args.clip_grad:
                total_norm = torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=1, norm_type=2) 

            trainer.step()
            
            training_loss = l.mean().cpu().item()
            print('global step: %s, epoch: %s, batch: %s, training loss: %.6f, time: %.2fs'
                % (global_step, epoch, batch, training_loss, time() - start_time), flush=True)
            
            batch += 1
            global_step += 1

        if args.depth_gcn:
            if isinstance(net, torch.nn.DataParallel):
                print("gcn weights:", net.module.STGN.gcn_weights_seg, net.module.STGN.gcn_weights_zone)
            else:
                print("gcn weights:", net.STGN.gcn_weights_seg, net.STGN.gcn_weights_zone)

        #compute va/test loss
        val_loss = compute_loss(net, val_loader, risk_mask, road_adj, risk_adj, poi_adj, global_step - 1, device, num_of_graph_node, args.inverse, scaler)
        print('global step: %s, epoch: %s, val loss：%.6f' % (global_step - 1, epoch, val_loss), flush=True)
        
        if args.use_nni:
            nni.report_intermediate_result(val_loss)

        if epoch == 1 or val_loss < early_stop.best_score:
            test_rmse, test_recall, test_map, test_inverse_trans_pre, test_inverse_trans_label = \
                        predict_and_evaluate(net, test_loader, risk_mask, road_adj, risk_adj, poi_adj, scaler, device, num_of_graph_node, inverse=args.inverse)

            high_test_rmse, high_test_recall, high_test_map, _, _ = \
                        predict_and_evaluate(net, high_test_loader, risk_mask, road_adj, risk_adj, poi_adj, scaler, device, num_of_graph_node, inverse=args.inverse)

            print('Segment global step: %s, epoch: %s, test RMSE: %.4f, test Recall: %.2f%%, test MAP: %.2f%%, high test RMSE: %.4f, high test Recall: %.2f%%, high test MAP: %.2f%%'
                % (global_step - 1, epoch, test_rmse[0], test_recall[0], test_map[0], high_test_rmse[0], high_test_recall[0], high_test_map[0]), flush=True)
            print('Zone global step: %s, epoch: %s, test RMSE: %.4f, test Recall: %.2f%%, test MAP: %.2f%%, high test RMSE: %.4f, high test Recall: %.2f%%, high test MAP: %.2f%%'
                % (global_step - 1, epoch, test_rmse[1], test_recall[1], test_map[1], high_test_rmse[1], high_test_recall[1], high_test_map[1]), flush=True)
        
        # early stop according to val loss
        early_stop(val_loss, test_rmse, test_recall, test_map,
                    high_test_rmse, high_test_recall, high_test_map,
                    test_inverse_trans_pre, test_inverse_trans_label) # 无需保留最佳模型！（test_rmse 的值可能是上一轮的，未更新！）

        if early_stop.early_stop:
            print('Segment best test RMSE: %.4f, best test Recall: %.2f%%, best test MAP: %.2f%%'
                % (early_stop.best_rmse[0], early_stop.best_recall[0], early_stop.best_map[0]), flush=True)
            print('Segment best test high RMSE: %.4f, best test high Recall: %.2f%%, best high test MAP: %.2f%%'
                    % (early_stop.best_high_rmse[0], early_stop.best_high_recall[0], early_stop.best_high_map[0]), flush=True)

            print('Zone best test RMSE: %.4f, best test Recall: %.2f%%, best test MAP: %.2f%%'
                    % (early_stop.best_rmse[1], early_stop.best_recall[1], early_stop.best_map[1]), flush=True)
            print('Zone best test high RMSE: %.4f, best test high Recall: %.2f%%, best high test MAP: %.2f%%'
                    % (early_stop.best_high_rmse[1], early_stop.best_high_recall[1], early_stop.best_high_map[1]), flush=True)
            break

    if not early_stop.early_stop:
        print("No Early Stopping")
        print('Segment best test RMSE: %.4f, best test Recall: %.2f%%, best test MAP: %.2f%%'
                % (early_stop.best_rmse[0], early_stop.best_recall[0], early_stop.best_map[0]), flush=True)
        print('Segment best test high RMSE: %.4f, best test high Recall: %.2f%%, best high test MAP: %.2f%%'
                % (early_stop.best_high_rmse[0], early_stop.best_high_recall[0], early_stop.best_high_map[0]), flush=True)

        print('Zone best test RMSE: %.4f, best test Recall: %.2f%%, best test MAP: %.2f%%'
                % (early_stop.best_rmse[1], early_stop.best_recall[1], early_stop.best_map[1]), flush=True)
        print('Zone best test high RMSE: %.4f, best test high Recall: %.2f%%, best high test MAP: %.2f%%'
                % (early_stop.best_high_rmse[1], early_stop.best_high_recall[1], early_stop.best_high_map[1]), flush=True)
    
    if args.use_nni:
        nni.report_final_result(early_stop.best_score)

    if args.check_result:
        if not args.use_nni:
            np.savez_compressed(os.path.join(local_dir, f"data/{data_type}_v5/pre_and_label_{data_type}_zone_v5_{config['decoder']}_check.npz"), pred_seg=early_stop.best_pre[0], label_seg=early_stop.best_label[0], pred_zone=early_stop.best_pre[1], label_zone=early_stop.best_label[1])

def main(config):
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    num_of_gru_layers = config['num_of_gru_layers']
    gru_hidden_size = config['gru_hidden_size']
    gcn_num_filter = config['gcn_num_filter']
    squeeze_dim =  config['squeeze_dim']
    encoder_layer = config['encoder_layer']
    n_head = config['n_head']
    dropout =  config['dropout']
    use_ASW = bool(config['use_ASW'])
    prefer_depth = bool(config['prefer_depth'])
    use_all_fea = bool(config["use_all_fea"])
    trans = bool(config["trans"])
    tkg = config["tkg"]
    num_experts = config["num_experts"]

    loss_weight = config["loss_weight"]
    component = config["component"]
    decoder = config["decoder"]

    if args.use_nni:
        batch_size = int(params['batch_size'])
        learning_rate = float(params['learning_rate'])
        num_of_gru_layers = int(params['num_of_gru_layers'])
        gru_hidden_size = int(params['gru_hidden_size'])
        gcn_num_filter = int(params['gcn_num_filter'])
        squeeze_dim = int(params["squeeze_dim"])
        encoder_layer = int(params['encoder_layer'])
        n_head = int(params['n_head'])
        num_experts = int(params['num_experts'])
        dropout =  float(params['dropout'])

        use_ASW = bool(params['use_ASW'])
        prefer_depth = bool(params['prefer_depth'])
        use_all_fea = bool(params["use_all_fea"])
        trans = bool(params["trans"])
        tkg = params["tkg"]

        loss_weight = float(params["loss_weight"])
        component = int(params["component"])
        decoder = params["decoder"]

    print("decoder", decoder)
    if decoder == "ori":
        args.use_sigmoid = bool(trans)
        args.inverse = False
    else:
        if args.inverse and args.use_sigmoid:
            p = np.random.rand()
            if p > 0.6:
                args.inverse = False
            else:
                args.use_sigmoid = False

    if args.check_result:
        print("compare label with preddiction")

    print(f"decoder is {decoder} with loss weight {loss_weight}")
    if args.inverse:
        print("Inverse transform the model output")

    if use_ASW:
        print("use adaptive sample weight")
    else:
        print("don't use adaptive sample weight")
    
    if  prefer_depth:
        print("prefer to depth representation")
    else:
        print("prefer to shallow representation")

    if use_all_fea:
        print("use all features")
    else:
        print("use two features (risk and flow)")

    if args.clip_grad:
        print("clip grad norm")

    if trans:
        print("transform the origin feature")
    else:
        print("don't transform the origin feature")

    if config["data_type"] == "brooklyn" or config["data_type"] == "manhatton":
        risk_mask_seg = np.load(mask_filename_seg, allow_pickle=True)["top_mask"]
        if args.no_top_mask:
            print("No top mask")
            risk_mask_seg = np.ones(risk_mask_seg.shape)
        if args.part_top_mask:
            print("Part top mask")
            risk_mask_seg = np.zeros(risk_mask_seg.shape)

            risk_flow = np.load(all_data_filename_seg, allow_pickle=True)["dataset"]

            risk_id = np.where(risk_flow[:, :, 0] > 1.0, 1, 0).sum(axis=0)
            print(risk_id.shape, risk_id) 

            threshold = 50
            if config['data_type'] == "brooklyn":
                threshold = 64
            for seg, val in enumerate(risk_id):
                if val > threshold:
                    risk_mask_seg[seg] = 1

            print("part top mask:", risk_mask_seg.sum() / risk_mask_seg.shape[0] * 100, f"% with {threshold}")

        zone2seg = np.load(zone2segment_filename, allow_pickle=True)["zone_adj_segment"]
        print("zone2seg:", zone2seg.sum(), zone2seg.shape)

        risk_mask_zone = np.dot(zone2seg, np.expand_dims(risk_mask_seg, axis=1)).squeeze()
        risk_mask_zone = np.where(risk_mask_zone > 0, 1.0, 0.0)

        print("zone_mask_seg:", risk_mask_zone.sum(), risk_mask_zone.shape)
        print("risk_mask_seg:", risk_mask_seg.sum(), risk_mask_seg.shape)

    print("risk_mask segment: ", risk_mask_seg.shape, risk_mask_seg.sum(), risk_mask_seg.max(), risk_mask_seg.min(), risk_mask_seg, flush=True)
    print("risk_mask zone: ", risk_mask_zone.shape, risk_mask_zone.sum(), risk_mask_zone.max(), risk_mask_zone.min(), risk_mask_zone, flush=True)

    train_loader, val_loader, test_loader, high_test_loader, scaler, num_of_graph_node, poi_feature, time_shape = generate_dataset(
                                    [all_data_filename_seg, all_data_filename_zone],
                                    train_rate=train_rate,
                                    valid_rate=valid_rate,
                                    recent_prior=recent_prior,
                                    week_prior=week_prior,
                                    one_day_period=one_day_period,
                                    days_of_week=days_of_week,
                                    pre_len=pre_len,
                                    risk_mask=[risk_mask_seg, risk_mask_zone],
                                    dataset=config["data_type"],
                                    augment_data=args.aug,
                                    adj_filename=[adj_filename_seg, adj_filename_zone],
                                    k=args.k_neighbors,
                                    batch_size=batch_size
    )

    nums_of_filter = []
    for _ in range(2):
        nums_of_filter.append(gcn_num_filter)

    depth_adjs_path = None
    kg_model_path = None
    if args.pretrain_kg:
        if config['data_type'] == 'brooklyn':
            model_path = "100_1024_0.001_0.0_68_500_0.4_32_10_0.68_v5_" + tkg + ".chkpnt"
        if config['data_type'] == 'manhatton':
            model_path = "100_64_0.001_0.0_68_500_0.4_32_10_0.68_v5_" + tkg + ".chkpnt"

        print(f"use tkg model {model_path}")

        depth_adjs_path = f"data/{config['data_type']}/tkg/depth_adjs_v5_{tkg}.npz"
        kg_model_path = f"data/{config['data_type']}/tkg/{model_path}"

        if args.use_nni:
            depth_adjs_path = os.path.join(nni_dir, depth_adjs_path)
            kg_model_path = os.path.join(nni_dir, kg_model_path)
        else:
            depth_adjs_path = os.path.join(local_dir, depth_adjs_path)
            kg_model_path = os.path.join(local_dir, kg_model_path)

    zone2seg = torch.from_numpy(zone2seg).to(torch.float32)
    poi_feature_seg = torch.from_numpy(poi_feature[0])
    poi_feature_zone = torch.from_numpy(poi_feature[1])
    if config['data_type'] == 'brooklyn':
        feature_dims = 2+17
        share_feaure = 53+32+26
    elif config['data_type'] == 'manhatton':
        feature_dims = 2+18
        share_feaure = 51+32+26
    STKGRisk_Model = STKGRisk(num_of_gru_layers, seq_len, pre_len,
                        gru_hidden_size, time_shape[1], feature_dims, share_feaure,
                        nums_of_filter, num_of_graph_node, squeeze_dim=squeeze_dim, component=component, zone2seg=zone2seg,
                        transform=trans, decoder=decoder, inverse=args.inverse, scaler=scaler,loss_weight=loss_weight,complicate=args.complicate,
                        use_sigmoid=args.use_sigmoid, prefer_depth=prefer_depth, depth_gcn=args.depth_gcn, pretrain_kg=args.pretrain_kg, encoder_layer=encoder_layer, dropout=dropout, n_head=n_head,
                        depth_adjs_path=depth_adjs_path, kg_model_path=kg_model_path, 
                        poi_feature=[poi_feature_seg, poi_feature_zone], use_all_fea=use_all_fea, num_experts=num_experts)

    # multi gpu
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!",flush=True)
        STKGRisk_Model = nn.DataParallel(STKGRisk_Model)
    STKGRisk_Model.to(device)
    print("Mixtrue model", STKGRisk_Model)

    num_of_parameters = 0
    for _, parameters in STKGRisk_Model.named_parameters():
        num_of_parameters += np.prod(parameters.shape)
    print("Number of Parameters: {}".format(num_of_parameters), flush=True)

    trainer = optim.Adam(STKGRisk_Model.parameters(), lr=learning_rate)
    early_stop = EarlyStopping(patience=patience, delta=delta)
    
    adjs = np.load(adj_filename_seg, allow_pickle=True)
    road_adj_seg = adjs["segment_adj"]
    poi_adj_seg = adjs["poi_adj"]
    risk_adj_seg = adjs["risk_adj"]
    risk_adj_seg_flag = np.isnan(risk_adj_seg)
    print("NAN cnt:", risk_adj_seg_flag.sum())
    risk_adj_seg = np.where(risk_adj_seg_flag, 0, risk_adj_seg)

    adjs = np.load(adj_filename_zone, allow_pickle=True)
    zone_adj = adjs["zone_adj"]
    poi_adj_zone = adjs["poi_adj_zone"]
    risk_adj_zone = adjs["risk_adj_zone"]
    
    training(
            STKGRisk_Model,
            training_epoch,
            train_loader,
            val_loader,
            test_loader,
            high_test_loader,
            [road_adj_seg, zone_adj],
            [risk_adj_seg, risk_adj_zone],
            [poi_adj_seg, poi_adj_zone],
            [risk_mask_seg, risk_mask_zone],
            trainer,
            early_stop,
            device,
            scaler,
            num_of_graph_node,
            data_type = config['data_type']
    )

if __name__ == "__main__":
    main(config)
