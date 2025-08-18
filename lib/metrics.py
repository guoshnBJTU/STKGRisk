from http.server import ThreadingHTTPServer
from xml.etree.ElementTree import register_namespace
import numpy as np
import math
import random

def transfer_dtype(y_true, y_pred):
    return y_true.astype('float32'), y_pred.astype('float32')

def mask_mse_np(y_true, y_pred, region_mask, null_val=None, check=False):
    """
    Arguments:
        y_true {np.ndarray} -- shape (samples, pre_len, N)
        y_pred {np.ndarray} -- shape (samples, pre_len, N)
        region_mask {np.ndarray} -- mask matrix, shape (N)
    
    Returns:
        np.float32 -- MSE
    """
    y_true, y_pred = transfer_dtype(y_true, y_pred)

    if not check:
        print("COR RMSE")
        y_true_ = []
        y_pred_ = []
        zero_sample = 0
        for i in range(y_true.shape[0]):
            if (y_true[i] * region_mask).sum() == 0.0:
                zero_sample += 1
                continue
            y_true_.append(y_true[i])
            y_pred_.append(y_pred[i])
        print("zero_sample(normal) ", y_true.shape, zero_sample)
        y_true = np.concatenate(y_true_, axis=0)
        y_pred = np.concatenate(y_pred_, axis=0)
        print("new shape(normal)", y_true.shape, y_pred.shape)
    else:
        print("COR RMSE")
        y_true_ = []
        zero_sample = 0
        for i in range(y_true.shape[0]):
            if (y_true[i] * region_mask).sum() == 0.0:
                zero_sample += 1
                continue
            y_true_.append(y_true[i])
        print("zero_sample(check) ", y_true.shape, zero_sample)
        y_true = np.concatenate(y_true_, axis=0)
        print("new shape(check)", y_true.shape)

    mask = region_mask

    mask = mask / mask.mean()

    return np.mean(mask * (y_true - y_pred) ** 2) # correct

def mask_rmse_np(y_true, y_pred, region_mask, null_val=None, check=False):
    """
    Arguments:
        y_true {np.ndarray} -- shape (samples, pre_len, N)
        y_pred {np.ndarray} -- shape (samples, pre_len, N)
        region_mask {np.ndarray} -- mask matrix, shape (N)
    
    Returns:
        np.float32 -- RMSE
    """
    y_true, y_pred = transfer_dtype(y_true, y_pred)
    return math.sqrt(mask_mse_np(y_true, y_pred, region_mask, null_val, check=check))

def nonzero_num(y_true, threshold=0.0):
    """get the grid number of have traffic accident in all time interval
    
    Arguments:
        y_true {np.array} -- shape:(samples, pre_len, N)
    Returns:
        {list} -- (samples, )
    """
    nonzero_list = []
    for i in range(len(y_true)):
        non_zero_nums = (y_true[i] > threshold).sum()
        nonzero_list.append(non_zero_nums)
    return nonzero_list

def get_top(data, accident_nums, threshold=0.0):
    """get top-K risk grid
    Arguments:
        data {np.array} -- shape (samples, pre_len, N)
        accident_nums {list} -- (samples, )，grid number of have traffic accident in all time intervals
    Returns:
        {list} -- (samples, k)
    """
    data = data.reshape((data.shape[0], -1)) # (samples * pre_len, N)
    topk_list = []
    high_risk_time_slie_cnt = 0
    for i in range(len(data)):
        risk = {}
        for j in range(len(data[i])): # N
            if data[i][j] > threshold:
                risk[j] = data[i][j]
        k = int(accident_nums[i])
        if len(risk) >= k:
            high_risk_time_slie_cnt += 1
            topk_list.append(list(dict(sorted(risk.items(), key=lambda x: x[1], reverse=True)[:k]).keys())) # id
        else:
            topk_list.append(list(dict(sorted(risk.items(), key=lambda x: x[1], reverse=True)).keys()))

    return topk_list


def Recall(y_true, y_pred, region_mask, check=False, threshold=0.0, topk=10, is_zone=False):
    """
    Arguments:
        y_true {np.ndarray} -- shape (samples, pre_len, N)
        y_pred {np.ndarray} -- shape (samples, pre_len, N)
        region_mask {np.ndarray} -- mask matrix, shape (N)
    Returns:
        float -- recall
    """
    tag = "segment"
    if is_zone:
        tag = "zone"

    region_mask = np.where(region_mask >= 1, 0, -1000)
    tmp_y_true = y_true + region_mask
    tmp_y_pred = y_pred + region_mask

    accident_grids_nums = nonzero_num(tmp_y_true, threshold)
    
    true_top_k = get_top(tmp_y_true, accident_grids_nums)
    if check and y_pred.min() == y_pred.max():
        pred_top_k = [random.sample(list(np.where(region_mask==0)[0]), i) for i in accident_grids_nums]
    else:
        if check:
            tmp_y_pred = np.repeat(np.expand_dims(tmp_y_pred, axis=[0, 1]), tmp_y_true.shape[0], axis=0)
        pred_top_k = get_top(tmp_y_pred, [topk] * y_true.shape[0], threshold=threshold)

    hit_sum = 0
    recall_cor = []
    for i in range(len(true_top_k)):
        intersection = [v for v in true_top_k[i] if v in pred_top_k[i]]
        hit_sum += len(intersection)
        if len(true_top_k[i]) > 0:
            recall_cor.append(len(intersection) / len(true_top_k[i]))
    
    if len(recall_cor) == 0:
        return 0
    else:
        return np.array(recall_cor).mean() * 100

all_or_high_seg = set()
all_or_high_zone = set()
val_dis = True
def MAP(y_true, y_pred, region_mask, check=False, test=True, threshold=0.0, topk=10, is_zone=False):
    """
        y_true {np.ndarray} -- shape (samples, pre_len, N) -> pre_len=1
        y_pred {np.ndarray} -- shape (samples, pre_len, N)
        region_mask {np.ndarray} -- mask matrix, shape (N)
    """
    tag = "segment"
    if is_zone:
        tag = "zone"

    region_mask = np.where(region_mask >= 1, 0, -1000)
    tmp_y_true = y_true + region_mask
    tmp_y_pred = y_pred + region_mask
    
    accident_grids_nums = nonzero_num(tmp_y_true, threshold)
    
    true_top_k = get_top(tmp_y_true, accident_grids_nums)
    if check and y_pred.min() == y_pred.max():
        pred_top_k = [random.sample(list(np.where(region_mask==0)[0]), i) for i in accident_grids_nums]
    else:
        if check:
            tmp_y_pred = np.repeat(np.expand_dims(tmp_y_pred, axis=[0, 1]), tmp_y_true.shape[0], axis=0)
        pred_top_k = get_top(tmp_y_pred, [topk] * y_true.shape[0], threshold=threshold)

    all_k_AP = []
    for sample in range(len(true_top_k)):
        if len(true_top_k[sample]) > 0:
            all_k_AP.append(AP(list(true_top_k[sample]), list(pred_top_k[sample])))
    
    if len(all_k_AP) == 0:
        return 0
    else:
        return sum(all_k_AP) / len(all_k_AP)

def AP(label_list, pre_list):
    hits = 0
    sum_precs = 0
    for n in range(len(pre_list)):
        if pre_list[n] in label_list: # pre_list[n] not in label_list: sum_precs += 0
            hits += 1
            sum_precs += hits / (n + 1.0)
    if hits > 0:
        return sum_precs / hits
    else:
        return 0

def mask_evaluation_np(y_true, y_pred, region_mask, null_val=None, test=True, is_zone=False):
    """RMSE，Recall，MAP
    
    Arguments:
        y_true {np.ndarray} -- shape (samples, pre_len, N)
        y_pred {np.ndarray} -- shape (samples, pre_len, N)
        region_mask {np.ndarray} -- mask matrix, shape (N)
    Returns:
        np.float32 -- MAE, MSE, RMSE
    """
    rmse_ = mask_rmse_np(y_true, y_pred, region_mask, null_val=null_val)

    if region_mask.shape[0] > 1000:
        threshold = 0.32 # bro
    else:
        threshold = 0.5 # man
    topk = 50
    if is_zone:
        if region_mask.shape[0] == 61:
            threshold = 2.0 # bro
        else:
            threshold = 1.0 # man
        topk = 20

    recall_ = Recall(y_true, y_pred, region_mask, threshold=threshold, topk=topk, is_zone=is_zone)
    map_ = MAP(y_true, y_pred, region_mask, threshold=threshold, test=test, topk=topk, is_zone=is_zone)

    return rmse_, recall_, map_ * 100