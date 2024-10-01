import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import xgboost as xgb
from train_pred_version import get_param_info

sys.path.append('..')
from model.FFN import FFN



def _param_path(base):
    return os.path.join(base, "parameter_new.json")

def _meta_path(base):
    return os.path.join(base, "meta_data.json")


def _cost_path(base):
    return os.path.join(base, "latency_matrix_pg_200.json")

def preprocess_z(vector: list,
                 params: list,
                 preprocessing_infos: list) -> torch.Tensor:
    """Generates a preprocessed vector for a given parameter input.

    input example
    vector = torch.tensor([25, 5.0, "apple"])
    params = [
        {"data_type": "int", "min": 0, "max": 100},
        {"data_type": "float"},
        {"data_type": "text", "distinct_values": ["apple", "banana", "cherry"]}
    ]
    preprocessing_infos = [
        {"type": "one_hot", "max_len" : 50},
        {"type": "std_normalization", "mean": 0.0, "variance": 1.0},
        {"type": "embedding", "output_dim": 5, "max_len" : 50}
    ]

    output example
    np.array([25, 0.6, 0])

    """

    processed_components = []
    vector = list(zip(*vector))
    for i, (param, preprocessing_info) in enumerate(zip(params, preprocessing_infos)):
        data_type = param["data_type"]
        preprocessing_type = preprocessing_info["type"]
        layer = vector[i]
        if data_type == "float" and preprocessing_type == "std_normalization":
            mean = preprocessing_info["mean"]
            std = torch.sqrt(preprocessing_info["variance"])
            processed_components.append((np.array(layer).astype(int) - mean) / std)
        elif data_type == "int":
            # shifted_layer = np.array(layer).astype(int) - param["min"]
            # processed_components.append(shifted_layer)
            if preprocessing_type == "embedding":
                vocab = {word: idx for idx, word in enumerate(param["distinct_values"])}
                num_oov_indices = preprocessing_info.get("num_oov_indices", 0)
                lookup_layer = np.array([vocab.get(la, len(vocab)) for la in layer])
                processed_components.append(lookup_layer)
            # elif preprocessing_type == "one_hot":
            #     processed_components.append(
            #         F.one_hot(shifted_layer.long(), num_classes=param["max"] - param["min"] + 1).float())

        elif data_type == "text":
            vocab = {word: idx for idx, word in enumerate(param["distinct_values"])}
            num_oov_indices = preprocessing_info.get("num_oov_indices", 0)
            lookup_layer = np.array([vocab.get(la, len(vocab)) for la in layer])
            processed_components.append(lookup_layer)
            # lookup_layer = torch.tensor(vocab.get(layer.item(), len(vocab)))
            # if preprocessing_type == "embedding":
            #     embed = nn.Embedding(len(vocab) + num_oov_indices,
            #                          preprocessing_info["output_dim"])
            #     processed_components.append(embed(lookup_layer))
            # elif preprocessing_type == "one_hot":
            #     processed_components.append(F.one_hot(lookup_layer, num_classes=len(vocab) + num_oov_indices).float())
        else:
            raise ValueError(f"Unsupported preprocessing: parameter type: {data_type}"
                             f" preprocessing type: {preprocessing_type}")

    ## Modified by zy, return the index of embedding/one-hot
    ## instead of the full vector

    # Concatenate all processed components into a single vector
    return np.transpose(np.array(processed_components))


def load_data(training_data, folder):
    path = os.path.join(training_data, folder)
    Z, latency, Z_t, latency_t, params, preprocess_info = [], [], [], [], [], []
    random.seed(42)

    with open(_param_path(path), 'r') as f:
        param = json.load(f)

    with open(_cost_path(path), 'r') as f:
        cost = json.load(f)

    with open(_meta_path(path), 'r') as f:
        meta = json.load(f)

    train_keys_list = list(cost.keys())
    test_keys_list = list(cost.keys())[160:]

    for param_key in train_keys_list:
        Z.append(list(param[param_key]))
        _, l = list(cost[param_key].items())[0]
        latency.append(float(l)/10000)

    for param_key in test_keys_list:
        Z_t.append(list(param[param_key]))
        _, l = list(cost[param_key].items())[0]
        latency_t.append(float(l)/10000)

    params, preprocess_info = get_param_info(meta)

    return Z, latency, Z_t, latency_t, params, preprocess_info


def train_XGBoost(Z, latency, model_path_file):
    # 准备数据
    dtrain = xgb.DMatrix(Z, label=latency)

    # 设置XGBoost的参数
    params = {
        'max_depth': 3,  # 深度较小
        'eta': 0.1,  # 学习率
        'objective': 'reg:squarederror',  # 回归任务
        'eval_metric': 'rmse'  # 均方根误差作为评估指标
    }
    num_rounds = 100  # 树的数量，需要调整以控制模型大小

    # 训练模型
    bst = xgb.train(params, dtrain, num_rounds)

    # 保存模型
    bst.save_model(f'{model_path_file}/xgb_model.model')

    # 检查模型大小是否接近16KB
    model_size = os.path.getsize(f'{model_path_file}/xgb_model.model')
    print("Model size: {} KB".format(model_size // 1024))
    return model_size // 1024


def training_share(training_data, model_path, device, epochs, data_size):
    all_folders = []
    for subdir, _, files in os.walk(training_data):
        if 'a' in os.path.basename(subdir) and "meta_data.json" in files:
            all_folders.append(os.path.basename(subdir))


def training_no_share_XGBoost(training_data, model_path, device):
    all_folders = []
    for subdir, _, files in os.walk(training_data):
        if 'a' in os.path.basename(subdir) and "meta_data.json" in files:
            all_folders.append(os.path.basename(subdir))

    print(all_folders)
    io_time = 0
    training_time = 0
    size = 0

    for folder in all_folders:
        print(f"Training for folder: {folder}")
        model_path_file = os.path.join(model_path, folder)
        os.makedirs(f'{model_path_file}', exist_ok=True)

        t1 = time.time()
        Z, latency, Z_t, latency_t, params, preprocess_info = load_data(training_data, folder)
        t2 = time.time()

        Z = preprocess_z(Z, params, preprocess_info)

        size += train_XGBoost(Z, latency, model_path_file)

        print(f"Finished training for folder: {folder}")

        t3 = time.time()

        io_time += (t2 - t1)
        training_time += (t3 - t2)
    print(f'size: {size}')

def training_no_share(training_data, model_path, device, epochs):
    all_folders = []
    for subdir, _, files in os.walk(training_data):
        if 'a' in os.path.basename(subdir) and "meta_data.json" in files:
            all_folders.append(os.path.basename(subdir))

    print(all_folders)
    io_time = 0
    training_time = 0
    pre_trained = 0

    for folder in all_folders:
        print(f"Training for folder: {folder}")
        model_path_file = os.path.join(model_path, folder)

        t1 = time.time()
        Z, latency, Z_t, latency_t, params, preprocess_info = load_data(training_data, folder)
        t2 = time.time()

        Z = preprocess_z(Z, params, preprocess_info)

        FFN_model = FFN(folder, preprocess_info, device=device)

        FFN_model.fit(Z, latency, pre_trained, 16, epochs)

        FFN_model.save(model_path_file)

        print(f"Finished training for folder: {folder}")

        t3 = time.time()

        io_time += (t2 - t1)
        training_time += (t3 - t2)


st = time.time()
training_no_share_XGBoost('../training_data/JOB/', '../checkpoints/xgb/', 'cpu')
et = time.time()
print(et-st)

# st = time.time()
# training_no_share('../training_data/JOB/', '../checkpoints/ffn/', 'cuda:1', 10)
# et = time.time()
# print(et-st)