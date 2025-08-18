from attr import attributes
import numpy as np
import sys
import os
from tqdm import *
import torch.utils.data as Data
import torch

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from lib.utils import Scaler_Bro_and_Man, min_max_transform

# high frequency time
high_fre_hour = [13, 14, 15, 16, 17, 18]

def generate_dataset(
                                    all_data_filename,
                                    train_rate=0.6,
                                    valid_rate=0.3,
                                    recent_prior=3,
                                    week_prior=4,
                                    one_day_period=24,
                                    days_of_week=7,
                                    pre_len=1,
                                    risk_mask=None,
                                    dataset="manhatton",
                                    augment_data=False,
                                    adj_filename=None,
                                    k=None,
                                    batch_size=4
):
    scaler_list = []
    num_of_graph_node = []
    time_shape = None
    poi_feature_list = []

    loaders_seg = []
    high_test_loaders_seg = []
    for idx, (x, y, target_times, x_factor, x_time, x_weather, high_x, high_y, high_target_times, high_x_factor, high_x_time, high_x_weather, poi_feature, scaler) in enumerate(normal_and_generate_dataset_time(
                                    all_data_filename[0],
                                    train_rate=train_rate,
                                    valid_rate=valid_rate,
                                    recent_prior=recent_prior,
                                    week_prior=week_prior,
                                    one_day_period=one_day_period,
                                    days_of_week=days_of_week,
                                    pre_len=pre_len,
                                    risk_mask=risk_mask[0],
                                    dataset=dataset,
                                    augment_data=augment_data,
                                    adj_filename=adj_filename[0],
                                    k=k,
                                    is_zone=0)):
        print(x.shape, y.shape)

        graph_x = x[:, :, [0, 1], :].reshape((x.shape[0], x.shape[1], 2, -1))
        high_graph_x = high_x[:, :, [0, 1], :].reshape((high_x.shape[0], high_x.shape[1], 2, -1))

        print("feature:", str(x.shape), "label:", str(y.shape), "time:", str(target_times.shape),
            "high feature:", str(high_x.shape), "high label:", str(high_y.shape))
        print("graph_x:", str(graph_x.shape), "high_graph_x:", str(high_graph_x.shape))

        if idx == 0:
            scaler_list.append(scaler)
            time_shape = target_times.shape
            num_of_graph_node.append(graph_x.shape[3])
            poi_feature_list.append(poi_feature)
        
        loaders_seg.append([target_times, graph_x, y, x_factor, x_time, x_weather])
        
        if idx == 2:
            high_test_loaders_seg.append([high_target_times, high_graph_x, high_y, high_x_factor, high_x_time, high_x_weather])

    loaders_zone = []
    high_test_loaders_zone = []
    for idx, (x, y, target_times, x_factor, x_time, x_weather, high_x, high_y, high_target_times, high_x_factor, high_x_time, high_x_weather, poi_feature, scaler) in enumerate(normal_and_generate_dataset_time(
                                    all_data_filename[1],
                                    train_rate=train_rate,
                                    valid_rate=valid_rate,
                                    recent_prior=recent_prior,
                                    week_prior=week_prior,
                                    one_day_period=one_day_period,
                                    days_of_week=days_of_week,
                                    pre_len=pre_len,
                                    risk_mask=risk_mask[1],
                                    dataset=dataset,
                                    augment_data=augment_data,
                                    adj_filename=adj_filename[1],
                                    k=k,
                                    is_zone=1)):
        print(x.shape, y.shape)

        graph_x = x[:, :, [0, 1], :].reshape((x.shape[0], x.shape[1], 2, -1))
        high_graph_x = high_x[:, :, [0, 1], :].reshape((high_x.shape[0], high_x.shape[1], 2, -1))

        print("feature:", str(x.shape), "label:", str(y.shape), "time:", str(target_times.shape),
            "high feature:", str(high_x.shape), "high label:", str(high_y.shape))
        print("graph_x:", str(graph_x.shape), "high_graph_x:", str(high_graph_x.shape))

        if idx == 0:
            scaler_list.append(scaler)
            num_of_graph_node.append(graph_x.shape[3])
            poi_feature_list.append(poi_feature)
        
        loaders_zone.append([target_times, graph_x, y, x_factor, x_time, x_weather])
        
        if idx == 2:
            high_test_loaders_zone.append([high_target_times, high_graph_x, high_y, high_x_factor, high_x_time, high_x_weather])

    loaders = []
    for idx, (seg, zone) in enumerate(zip(loaders_seg, loaders_zone)):
        target_times = seg[0]

        graph_x = np.concatenate([seg[1], zone[1]], axis=3)
        y = np.concatenate([seg[2], zone[2]], axis=2)
        print("After concatenate", graph_x.shape, y.shape)

        x_factor = seg[3]
        x_time = seg[4]
        x_weather = seg[5]

        loaders.append(Data.DataLoader(
            Data.TensorDataset(
                torch.from_numpy(target_times).to(torch.float32),
                torch.from_numpy(graph_x).to(torch.float32), # RuntimeError: expected scalar type Double but found Float
                torch.from_numpy(y).to(torch.float32),
                torch.from_numpy(x_factor).to(torch.float32),
                torch.from_numpy(x_time).to(torch.float32),
                torch.from_numpy(x_weather).to(torch.float32)
                ),
            batch_size=batch_size,
            shuffle=(idx == 0)
        ))
    
    high_test_loader = None
    for (seg, zone) in zip(high_test_loaders_seg, high_test_loaders_zone):
        target_times = seg[0]

        graph_x = np.concatenate([seg[1], zone[1]], axis=3)
        y = np.concatenate([seg[2], zone[2]], axis=2)
        print("After concatenate (high)", graph_x.shape, y.shape)

        x_factor = seg[3]
        x_time = seg[4]
        x_weather = seg[5]
        # print(x_factor.shape, x_time.shape, x_weather.shape)

        high_test_loader = Data.DataLoader(
            Data.TensorDataset(
                torch.from_numpy(target_times).to(torch.float32),
                torch.from_numpy(graph_x).to(torch.float32), # RuntimeError: expected scalar type Double but found Float
                torch.from_numpy(y).to(torch.float32),
                torch.from_numpy(x_factor).to(torch.float32),
                torch.from_numpy(x_time).to(torch.float32),
                torch.from_numpy(x_weather).to(torch.float32)
                ),
            batch_size=batch_size,
            shuffle=True
        )

    train_loader, val_loader, test_loader = loaders

    return train_loader, val_loader, test_loader, high_test_loader, scaler_list, num_of_graph_node, poi_feature_list, time_shape

def split_and_norm_data_time_bro_and_man(all_data, 
                        train_rate = 0.6, 
                        valid_rate = 0.2, 
                        recent_prior=3, 
                        week_prior=4, 
                        one_day_period=24, 
                        days_of_week=7, 
                        pre_len=1,
                        risk_mask=None,
                        augment_data=False,
                        adj_filename=None,
                        k=3,
                        is_zone=0):
    if is_zone == 1:
        risk_flow = all_data["zone_dataset"]
        all_poi_feature = np.transpose(all_data["poi_feature_zone"], (0, 2, 1)) # (T'=1, N, D'=17) -> (T, D', N')
    else:
        risk_flow = all_data["dataset"]
        all_poi_feature = np.transpose(all_data["poi_feature"], (0, 2, 1)) # (T'=1, N, D'=17) -> (T, D', N')
    all_factor_feature = np.transpose(all_data["factor_feature"], (0, 2, 1)) # (T, N'=1, D'=53) -> (T, D', N')
    all_time_feature = np.transpose(all_data["time_feature"], (0, 2, 1)) # (T, N'=1, D'=32) -> (T, D', N')
    all_weather_feature = np.transpose(all_data["weather_feature"], (0, 2, 1)) # (T, N'=1, D'=26) -> (T, D', N')
    print("external feature: ", all_factor_feature.shape, all_time_feature.shape, all_weather_feature.shape, all_poi_feature.shape)

    all_weather_feature[:, 25:, :] = min_max_transform(all_weather_feature[:, 25:, :])
    all_factor_feature = min_max_transform(all_factor_feature)
    all_poi_feature = min_max_transform(all_poi_feature)

    risk_flow = np.transpose(risk_flow, (0, 2, 1)) # (T, D, N)

    num_of_time, _, _ = risk_flow.shape
    train_line, valid_line = int(num_of_time * train_rate), int(num_of_time * (train_rate + valid_rate))
    for index, (start, end) in enumerate(((0, train_line), (train_line, valid_line), (valid_line, num_of_time))):
        if index == 0:
            scaler = Scaler_Bro_and_Man(risk_flow[start:end])

        norm_data = scaler.transform(risk_flow[start:end, :, :])

        factor_feature = all_factor_feature[start:end, :, 0]    
        time_feature = all_time_feature[start:end, :, 0]
        weather_feature = all_weather_feature[start:end, :, 0]

        poi_feature = all_poi_feature[0, :, :]
        print("external features:", factor_feature.shape, time_feature.shape, weather_feature.shape, poi_feature.shape)

        X, Y, target_time, x_factor, x_time, x_weather = [], [], [], [], [], []
        high_X, high_Y, high_target_time, high_x_factor, high_x_time, high_x_weather = [], [], [], [], [], []
        for i in tqdm(range(len(norm_data) - week_prior * days_of_week * one_day_period - pre_len + 1)):
            t = i + week_prior * days_of_week * one_day_period
            label = norm_data[t:t + pre_len, 0, :]
            period_list = []
            for week in range(week_prior):
                period_list.append(i + week * days_of_week * one_day_period)
            for recent in list(range(1, recent_prior + 1))[::-1]:
                period_list.append(t - recent) 

            feature = norm_data[period_list, :, :]
            X.append(feature)
            Y.append(label)
            
            x_factor.append(factor_feature[period_list])
            x_time.append(time_feature[period_list])
            x_weather.append(weather_feature[period_list])

            target_time.append(time_feature[t, :])

            if list(time_feature[t, :24]).index(1) in high_fre_hour:
                high_X.append(feature)
                high_Y.append(label)

                high_x_factor.append(factor_feature[period_list])
                high_x_time.append(time_feature[period_list])
                high_x_weather.append(weather_feature[period_list])

                high_target_time.append(time_feature[t, :])
        yield np.array(X), np.array(Y), np.array(target_time), np.array(x_factor), np.array(x_time), np.array(x_weather),\
            np.array(high_X), np.array(high_Y), np.array(high_target_time), np.array(high_x_factor), np.array(high_x_time), np.array(high_x_weather),\
            poi_feature, scaler

def normal_and_generate_dataset_time(
        all_data_filename, 
        train_rate=0.6, 
        valid_rate=0.2, 
        recent_prior=3, 
        week_prior=4, 
        one_day_period=24, 
        days_of_week=7, 
        pre_len=1,
        risk_mask=None,
        dataset="nyc",
        augment_data=False,
        adj_filename=None,
        k=3,
        is_zone=0):
    if dataset == "brooklyn" or dataset == "manhatton":
        all_data = np.load(all_data_filename, allow_pickle=True)

        for i in split_and_norm_data_time_bro_and_man(all_data, 
                        train_rate=train_rate, 
                        valid_rate=valid_rate, 
                        recent_prior=recent_prior, 
                        week_prior=week_prior, 
                        one_day_period=one_day_period, 
                        days_of_week=days_of_week, 
                        pre_len=pre_len,
                        risk_mask=risk_mask,
                        augment_data=augment_data,
                        adj_filename=adj_filename,
                        k=k,
                        is_zone=is_zone):
            yield i