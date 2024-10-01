import os
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
# from model.TreeConvolution.util import *

import joblib
from .feature import SampleEntity
from .TreeConvolution.tcnn import (BinaryTreeConv, DynamicPooling,
                                   TreeActivation, TreeLayerNorm)
from .TreeConvolution.util import prepare_trees

np.random.seed(42)

Template_DIM = []


def _nn_path(base):
    return os.path.join(base, "nn_weights")


def _fnn_path_first_layer(base, template_id):
    return os.path.join(base, "fnn_weights_" + template_id)


def _fnn_path(base):
    return os.path.join(base, "fnn_weights")


def _feature_generator_path(base):
    return os.path.join(base, "feature_generator")


def _input_feature_dim_path(base):
    return os.path.join(base, "input_feature_dim")

def _cost_est_net_path(base):
    return os.path.join(base, "cost_est_net")


def collate_fn(x):
    trees = []
    targets = []

    for tree, target in x:
        trees.append(tree)
        targets.append(target)

    targets = torch.tensor(targets)
    return trees, targets


class NonePairDataset(Dataset):
    def __init__(self, X, Y, Z):
        self.X = X
        self.Y = Y
        self.Z = Z

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Z[idx], self.Y[idx]


def generate_dataset(X1, X2, Y1, Y2, Z):
    assert len(X1) == len(X2) == len(Y1) == len(Y2) == len(Z)

    total_size = len(X1)
    indices = list(range(total_size))
    np.random.shuffle(indices)

    combined_X = [X1[i] for i in indices] + [X2[i] for i in indices]
    combined_Y = [Y1[i] for i in indices] + [Y2[i] for i in indices]
    combined_Z = [Z[i] for i in indices] + [Z[i] for i in indices]

    dataset = NonePairDataset(combined_X, combined_Y, combined_Z)

    return dataset


def split_dataset(X1, X2, Y1, Y2, Z, train_ratio=0.8, val_ratio=0.1):
    assert len(X1) == len(X2) == len(Y1) == len(Y2) == len(Z)

    total_size = len(X1)
    indices = list(range(total_size))
    np.random.shuffle(indices)  # shuffle indices randomly

    # Determine the split sizes
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # Split the indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Create sub-datasets
    train_dataset = generate_dataset([X1[i] for i in train_indices],
                                [X2[i] for i in train_indices],
                                [Y1[i] for i in train_indices],
                                [Y2[i] for i in train_indices],
                                [Z[i] for i in train_indices])

    val_dataset = generate_dataset([X1[i] for i in val_indices],
                              [X2[i] for i in val_indices],
                              [Y1[i] for i in val_indices],
                              [Y2[i] for i in val_indices],
                              [Z[i] for i in val_indices])

    test_dataset = generate_dataset([X1[i] for i in test_indices],
                               [X2[i] for i in test_indices],
                               [Y1[i] for i in test_indices],
                               [Y2[i] for i in test_indices],
                               [Z[i] for i in test_indices])

    return train_dataset, val_dataset, test_dataset


def collate_none_pairwise_fn(x):
    trees = []
    parameters = []
    labels = []

    for tree, parameter, label in x:
        trees.append(tree)
        parameters.append(parameter)
        labels.append(label)
    return trees, torch.FloatTensor(np.array(parameters)), torch.FloatTensor(np.array(labels)).reshape(-1, 1)


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
        one_hot.scatter_(1, x.long(), 1)
        return one_hot  # Batch x num_classes


class PlanEmbeddingNet(nn.Module):
    def __init__(self, input_feature_dim) -> None:
        super(PlanEmbeddingNet, self).__init__()
        self.input_feature_dim = input_feature_dim
        # self._cuda = False
        # self.device = None

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
        device = next(self.parameters()).device
        return prepare_trees(feature, transformer, left_child, right_child, device=device)

    def cuda(self, device):
        # self._cuda = True
        # self.device = device
        # return super().cuda()
        self.to(device)


class PlanEmbeddingNetPredVersion(nn.Module):
    def __init__(self, input_feature_dim, max_predicate_len = 30, max_col = 200, max_op = 20) -> None:
        super(PlanEmbeddingNetPredVersion, self).__init__()
        self.input_feature_dim = input_feature_dim
        self.max_predicate_len = max_predicate_len
        # self._cuda = False
        # self.device = None

        self.col_embed = nn.Embedding(max_col, 32)
        self.op_embed = nn.Embedding(max_op, 32)

        self.tree_conv = nn.Sequential(
            BinaryTreeConv(self.input_feature_dim + 64 - 2 * self.max_predicate_len - 1, 256),
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
        device = next(self.parameters()).device
        feature, indexes = trees
        # Batch x Feature_Dim x Nodes --> Batch x Nodes x Feature_Dim
        b, d, n = feature.size()
        feature = feature.transpose(1,2).view(b*n, d)

        other_len = self.input_feature_dim - 2 * self.max_predicate_len - 1
        others, columns, ops, length = \
            torch.split(feature,(other_len, \
                self.max_predicate_len,self.max_predicate_len,1), dim = -1)

        col_embed = self.col_embed(columns.long())
        ops_embed = self.op_embed(ops.long())
        concat = torch.cat((col_embed,ops_embed), dim = -1)

        mask = (torch.arange(self.max_predicate_len).expand(len(length), self.max_predicate_len) < length.to('cpu')).to(device)
        concat[~mask] = 0.

        total = torch.sum(concat, dim = 1)

        collate_feature = torch.cat((others, total), dim=-1).view(b, n, -1).transpose(1,2) # shift back

        # print(other_len, length)
        # print(collate_feature.size(), indexes.size(), others.size())
        return self.tree_conv((collate_feature, indexes))

    def build_trees(self, feature):
        device = next(self.parameters()).device
        return prepare_trees(feature, transformer, left_child, right_child, device=device)


class ParameterEmbeddingNet(nn.Module):
    def __init__(self, template_id, preprocessing_infos, embed_dim=16):
        super(ParameterEmbeddingNet, self).__init__()

        self.id = template_id
        self.embed_dim = embed_dim

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
                layers.append(nn.Embedding(info["max_len"], embed_dim))
                embed_len += embed_dim
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
                embedded.append(e(x_i.long()).view(batch_size, -1))
            else:
                embedded.append(e(x_i))

        embedded = torch.concat(embedded, -1)

        x = F.relu(self.fc1(embedded))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CostEstNet(nn.Module):
    def __init__(self, input_dim):
        super(CostEstNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RankPQOModel():
    def __init__(self, feature_generator, template_id, preprocessing_infos, device, is_predict=True) -> None:
        super(RankPQOModel, self).__init__()
        self._feature_generator = feature_generator
        self._input_feature_dim = None
        self._model_parallel = None
        self._template_id = template_id
        self.preprocessing_infos = preprocessing_infos
        self.device = device
        self.is_predict = is_predict

        self.columns = None
        self.ops = None

    def load(self, path, fist_layer=0):
        with open(_input_feature_dim_path(path), "rb") as f:
            self._input_feature_dim = joblib.load(f)

        if self.is_predict:
            self.plan_net = PlanEmbeddingNetPredVersion(self._input_feature_dim).to(self.device)
        else:
            self.plan_net = PlanEmbeddingNet(self._input_feature_dim).to(self.device)
        self.plan_net.load_state_dict(torch.load(
            _nn_path(path), map_location=torch.device(self.device)))
        self.plan_net.eval()

        self.parameter_net = ParameterEmbeddingNet(self._template_id, self.preprocessing_infos).to(self.device)
        if fist_layer:
            state_dicts = torch.load(_fnn_path_first_layer(path, self._template_id), map_location=torch.device(self.device))
            self.parameter_net.embed_layers.load_state_dict(state_dicts['embed_layer'])
            self.parameter_net.fc1.load_state_dict(state_dicts['fc1'])
        state_dicts = torch.load(_fnn_path(path), map_location=torch.device(self.device))
        self.parameter_net.fc2.load_state_dict(state_dicts['fc2'])
        self.parameter_net.fc3.load_state_dict(state_dicts['fc3'])
        self.parameter_net.eval()

        self.cost_est_net = CostEstNet(64).to(self.device)
        self.cost_est_net.load_state_dict(torch.load(_cost_est_net_path(path), map_location=torch.device(self.device)))
        self.cost_est_net.eval()

        with open(_feature_generator_path(path), "rb") as f:
            self._feature_generator = joblib.load(f)

    def save(self, path):
        os.makedirs(path, exist_ok=True)

        torch.save(self.plan_net.state_dict(), _nn_path(path))
        torch.save({
            'embed_layer': self.parameter_net.embed_layers.state_dict(),
            'fc1': self.parameter_net.fc1.state_dict()
        }, _fnn_path_first_layer(path, self._template_id))
        torch.save({
            'fc2': self.parameter_net.fc2.state_dict(),
            'fc3': self.parameter_net.fc3.state_dict()
        }, _fnn_path(path))

        torch.save(self.cost_est_net.state_dict(), _cost_est_net_path(path))

        with open(_feature_generator_path(path), "wb") as f:
            joblib.dump(self._feature_generator, f)
        with open(_input_feature_dim_path(path), "wb") as f:
            joblib.dump(self._input_feature_dim, f)

    def evaluate(self, dataset, dataloader):
        criterion = nn.MSELoss()

        self.plan_net.eval()
        self.parameter_net.eval()

        loss_accum = 0
        with torch.no_grad():
            for x, z, label in dataloader:

                z = z.to(self.device)
                label = label.to(self.device)

                tree_x = self.plan_net.build_trees(x)

                # pairwise
                x_pred = self.plan_net(tree_x)
                z_pred = self.parameter_net(z)
                prob_y = self.cost_est_net(torch.cat((x_pred, z_pred), dim=1)).float()

                loss = criterion(prob_y, label.float())
                loss_accum += loss.item()


        # Compute average loss
        avg_loss = loss_accum / len(dataset)

        # Return to training mode
        self.plan_net.train()
        self.parameter_net.train()

        return avg_loss

    def fit_with_test(self, X1, X2, Y1, Y2, Z, pre_training=False, batch_size=16, epochs=50):
        print(f"Lengths -> X1: {len(X1)}, X2: {len(X2)}, Y1: {len(Y1)}, Y2: {len(Y2)}, Z: {len(Z)}")
        assert len(X1) == len(X2) and len(Y1) == len(Y2) and len(X1) == len(Y1) and len(X1) == len(Z)
        if isinstance(Y1, list):
            Y1 = np.array(Y1)
            Y1 = Y1.reshape(-1, 1)
        if isinstance(Y2, list):
            Y2 = np.array(Y2)
            Y2 = Y2.reshape(-1, 1)

        # # determine the initial number of channels
        if not pre_training:
            # input_feature_dim = len(X1[0].get_feature())
            input_feature_dim = X1[0].get_feature_len()
            print("input_feature_dim:", input_feature_dim)

            if self.is_predict:
                self.plan_net = PlanEmbeddingNetPredVersion(input_feature_dim).to(self.device)
            else:
                self.plan_net = PlanEmbeddingNet(input_feature_dim).to(self.device)

            self._input_feature_dim = input_feature_dim
            self.parameter_net = ParameterEmbeddingNet(self._template_id, self.preprocessing_infos).to(self.device)

            self.cost_est_net = CostEstNet(64).to(self.device)

            self.plan_net.train()
            self.parameter_net.train()
            self.cost_est_net.train()

        # Splitting the dataset
        train_dataset, val_dataset, test_dataset = split_dataset(X1, X2, Y1, Y2, Z)

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      collate_fn=collate_none_pairwise_fn)
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    collate_fn=collate_none_pairwise_fn)
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     collate_fn=collate_none_pairwise_fn)

        plan_optimizer = torch.optim.Adam(self.plan_net.parameters())
        parameter_optimizer = torch.optim.Adam(self.parameter_net.parameters())
        cost_optimizer = torch.optim.Adam(self.cost_est_net.parameters())

        criterion = nn.MSELoss()

        losses = []
        start_time = time()
        for epoch in range(epochs):
            loss_accum = 0
            for x, z, label in train_dataloader:
                z = z.to(self.device)
                label = label.to(self.device)

                tree_x = self.plan_net.build_trees(x)

                # pairwise
                x_pred = self.plan_net(tree_x)
                z_pred = self.parameter_net(z)
                prob_y = self.cost_est_net(torch.cat((x_pred, z_pred), dim=1)).float()

                loss = criterion(prob_y, label.float())

                loss_accum += loss.item()

                plan_optimizer.zero_grad()
                parameter_optimizer.zero_grad()
                cost_optimizer.zero_grad()
                loss.backward()
                plan_optimizer.step()
                parameter_optimizer.step()
                cost_optimizer.step()

            loss_accum /= len(train_dataset)
            losses.append(loss_accum)
            print("Epoch", epoch, "training loss:", loss_accum)

            if (epoch + 1) % 5 == 0:
                loss = self.evaluate(val_dataset, val_dataloader)
                print("validation loss:", loss)

        print("training time:", time() - start_time, "batch size:", batch_size)
        loss = self.evaluate(test_dataset, test_dataloader)
        print("test loss:", loss)

    def test(self, X1, X2, Y1, Y2, Z, batch_size=16):
        assert len(X1) == len(X2) and len(Y1) == len(Y2) and len(X1) == len(Y1) and len(X1) == len(Z)
        if isinstance(Y1, list):
            Y1 = np.array(Y1)
            Y1 = Y1.reshape(-1, 1)
        if isinstance(Y2, list):
            Y2 = np.array(Y2)
            Y2 = Y2.reshape(-1, 1)

        # Splitting the dataset
        train_dataset, val_dataset, test_dataset = split_dataset(X1, X2, Y1, Y2, Z)

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      collate_fn=collate_none_pairwise_fn)
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    collate_fn=collate_none_pairwise_fn)
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     collate_fn=collate_none_pairwise_fn)

        loss = self.evaluate(train_dataset, train_dataloader)
        print("train loss:", loss)
        loss = self.evaluate(val_dataset, val_dataloader)
        print("validation loss:", loss)
        loss = self.evaluate(test_dataset, test_dataloader)
        print("test loss:", loss)

