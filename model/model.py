import os
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import joblib
from feature import SampleEntity
from TreeConvolution.tcnn import (BinaryTreeConv, DynamicPooling,
                                  TreeActivation, TreeLayerNorm)
from TreeConvolution.util import prepare_trees

CUDA = torch.cuda.is_available()
GPU_LIST = [0, 1]

torch.set_default_tensor_type(torch.DoubleTensor)
device = torch.device("cuda:0" if CUDA else "cpu")

Template_DIM = []


def _nn_path(base):
    return os.path.join(base, "nn_weights")


def _fnn_path(base, template_id):
    return os.path.join(base, "fnn_weights_" + template_id)


def _feature_generator_path(base):
    return os.path.join(base, "feature_generator")


def _input_feature_dim_path(base):
    return os.path.join(base, "input_feature_dim")


def collate_fn(x):
    trees = []
    targets = []

    for tree, target in x:
        trees.append(tree)
        targets.append(target)

    targets = torch.tensor(targets)
    return trees, targets


def collate_pairwise_fn(x):
    trees1 = []
    trees2 = []
    parameters2 = []
    labels = []

    for tree1, tree2, parameter2, label in x:
        trees1.append(tree1)
        trees2.append(tree2)
        parameters2.append(parameter2)
        labels.append(label)
    return trees1, trees2, parameters2, labels


def transformer(x: SampleEntity):
    return x.get_feature()


def left_child(x: SampleEntity):
    return x.get_left()


def right_child(x: SampleEntity):
    return x.get_right()


class OneHotNN(nn.Module):
    def __init__(self, num_classes):
        super(OneHotNN, self).__init__()
        self.num_classes = num_classes

    def forward(self, x):
        # Batch x 1
        one_hot = torch.zeros(x.size(0), self.num_classes, device=x.device)
        one_hot.scatter_(1, x, 1)
        return one_hot  # Batch x num_classes


class PlanEmbeddingNet(nn.Module):
    def __init__(self, input_feature_dim) -> None:
        super(PlanEmbeddingNet, self).__init__()
        self.input_feature_dim = input_feature_dim
        self._cuda = False
        self.device = None

        self.tree_conv = nn.Sequential(
            BinaryTreeConv(self.input_feature_dim, 256),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(256, 128),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(128, 64),
            TreeLayerNorm(),
            DynamicPooling(),
            nn.Linear(64, 32)
        )

    def forward(self, trees):
        return self.tree_conv(trees)

    def build_trees(self, feature):
        return prepare_trees(feature, transformer, left_child, right_child, cuda=self._cuda, device=self.device)

    def cuda(self, device):
        self._cuda = True
        self.device = device
        return super().cuda()


class ParameterEmbeddingNet(nn.Module):
    def __init__(self, template_id, preprocessing_infos):
        super(ParameterEmbeddingNet, self).__init__()

        self.id = template_id
        # input_dim = Template_DIM[template_id]
        # Layers

        layers = []
        self.length = len(preprocessing_infos)
        embed_len = 0

        for info in preprocessing_infos:
            if info["type"] == "one_hot":
                layers.append(OneHotNN(info['max_len']))
                embed_len += info['max_len']
            elif info["type"] == "std_normalization":
                layers.append(nn.Identity())
                embed_len += 1
            elif info["type"] == "embedding":
                layers.append(nn.Embedding(info["max_len"], info["output_dim"]))
                embed_len = info['output_dim']
            else:
                raise ValueError(f"Unknown preprocessing type: {info['type']}")
        self.embed_layers = nn.ModuleList(layers)
        self.embed_len = embed_len

        self.fc1 = nn.Linear(embed_len, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)

    def forward(self, x):
        ## x.shape : Batch x len(preprocessing_infos)
        batch_size = x.size(0)
        x_l = torch.split(x, 1, dim=-1)  # list of Batch x 1
        embedded = []
        for x_i, e in zip(x_l, self.embed_layers):
            if not isinstance(e, nn.Identity):
                embedded.append(e(x_i).long()).view(batch_size, -1)
            else:
                embedded.append(e(x_i))

        embedded = torch.concat(embedded, -1)

        x = F.relu(self.fc1(embedded))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RankPQOModel():
    def __init__(self, feature_generator, template_id) -> None:
        super(self).__init__()
        self._feature_generator = feature_generator
        self._input_feature_dim = None
        self._model_parallel = None
        self._template_id = template_id
        self._parameter_input_dim = Template_DIM[template_id]

    def load(self, path):
        with open(_input_feature_dim_path(path), "rb") as f:
            self._input_feature_dim = joblib.load(f)

        self.plan_net = PlanEmbeddingNet(self._input_feature_dim)
        if CUDA:
            self.plan_net.load_state_dict(torch.load(_nn_path(path)))
        else:
            self.plan_net.load_state_dict(torch.load(
                _nn_path(path), map_location=torch.device('cpu')))
        self.plan_net.eval()

        self.parameter_net = ParameterEmbeddingNet(self._parameter_input_dim)
        if CUDA:
            self.parameter_net.load_state_dict(torch.load(_fnn_path(path, self._template_id)))
        else:
            self.parameter_net.load_state_dict(torch.load(
                _fnn_path(path, self._template_id), map_location=torch.device('cpu')))
        self.parameter_net.eval()

        with open(_feature_generator_path(path), "rb") as f:
            self._feature_generator = joblib.load(f)

    def save(self, path):
        os.makedirs(path, exist_ok=True)

        if CUDA:
            torch.save(self.plan_net.module.state_dict(), _nn_path(path))
        else:
            torch.save(self.plan_net.state_dict(), _nn_path(path))

        if CUDA:
            torch.save(self.parameter_net.module.state_dict(), _fnn_path(path, self._template_id))
        else:
            torch.save(self.parameter_net.state_dict(), _fnn_path(path, self._template_id))

        with open(_feature_generator_path(path), "wb") as f:
            joblib.dump(self._feature_generator, f)
        with open(_input_feature_dim_path(path), "wb") as f:
            joblib.dump(self._input_feature_dim, f)

    def fit(self, X1, X2, Y1, Y2, Z, pre_training=False):
        assert len(X1) == len(X2) and len(Y1) == len(Y2) and len(X1) == len(Y1) and len(X1) == len(Z)
        if isinstance(Y1, list):
            Y1 = np.array(Y1)
            Y1 = Y1.reshape(-1, 1)
        if isinstance(Y2, list):
            Y2 = np.array(Y2)
            Y2 = Y2.reshape(-1, 1)

        # # determine the initial number of channels
        if not pre_training:
            input_feature_dim = len(X1[0].get_feature())
            print("input_feature_dim:", input_feature_dim)

            self.plan_net = PlanEmbeddingNet(input_feature_dim)
            self._input_feature_dim = input_feature_dim
            self.parameter_net = PlanEmbeddingNet(self._template_id)
            if CUDA:
                self.plan_net = self.plan_net.cuda(device)
                self.plan_net = torch.nn.DataParallel(
                    self.plan_net, device_ids=GPU_LIST)
                self.plan_net.cuda(device)
                self.parameter_net = self.parameter_net.cuda(device)
                self.parameter_net = torch.nn.DataParallel(
                    self.parameter_net, device_ids=GPU_LIST)
                self.parameter_net.cuda(device)

        pairs = []
        for i in range(len(X1)):
            pairs.append((X1[i], X2[i], Z[i], 1.0 if Y1[i] <= Y2[i] else 0.0))

        batch_size = 64
        if CUDA:
            batch_size = batch_size * len(GPU_LIST)

        dataset = DataLoader(pairs,
                             batch_size=batch_size,
                             shuffle=True,
                             collate_fn=collate_pairwise_fn)

        plan_optimizer = None
        parameter_optimizer = None
        if CUDA:
            plan_optimizer = torch.optim.Adam(self.plan_net.module.parameters())
            plan_optimizer = nn.DataParallel(plan_optimizer, device_ids=GPU_LIST)
            parameter_optimizer = torch.optim.Adam(self.parameter_net.parameters())
            parameter_optimizer = nn.DataParallel(parameter_optimizer, device_ids=GPU_LIST)
        else:
            plan_optimizer = torch.optim.Adam(self.plan_net.parameters())
            parameter_optimizer = torch.optim.Adam(self.parameter_net.parameters())

        bce_loss_fn = torch.nn.BCELoss()

        losses = []
        start_time = time()
        for epoch in range(100):
            loss_accum = 0
            for x1, x2, z, label in dataset:

                tree_x1, tree_x2 = None, None
                if CUDA:
                    tree_x1 = self.plan_net.module.build_trees(x1)
                    tree_x2 = self.plan_net.module.build_trees(x2)
                else:
                    tree_x1 = self.plan_net.build_trees(x1)
                    tree_x2 = self.plan_net.build_trees(x2)

                # pairwise
                y_pred_1 = self.plan_net(tree_x1)
                y_pred_2 = self.plan_net(tree_x2)
                z_pred = self.parameter_net(z)
                distance_1 = torch.norm(y_pred_1 - z_pred)
                distance_2 = torch.norm(y_pred_2 - z_pred)
                prob_y = 1.0 if distance_1 <= distance_2 else 0.0

                label_y = torch.tensor(np.array(label).reshape(-1, 1))
                if CUDA:
                    label_y = label_y.cuda(device)

                loss = bce_loss_fn(prob_y, label_y)
                loss_accum += loss.item()

                if CUDA:
                    plan_optimizer.module.zero_grad()
                    parameter_optimizer.module.zero_grad()
                    loss.backward()
                    plan_optimizer.module.step()
                    parameter_optimizer.module.step()
                else:
                    plan_optimizer.zero_grad()
                    parameter_optimizer.zero_grad()
                    loss.backward()
                    plan_optimizer.step()
                    parameter_optimizer.step()

            loss_accum /= len(dataset)
            losses.append(loss_accum)

            print("Epoch", epoch, "training loss:", loss_accum)
        print("training time:", time() - start_time, "batch size:", batch_size)
