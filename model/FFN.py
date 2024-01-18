import os
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import joblib

np.random.seed(42)

def _fnn_path_first_layer(base, template_id):
    return os.path.join(base, "FFN/fnn_weights_" + template_id)


def _fnn_path(base):
    return os.path.join(base, "FFN/fnn_weights")


def _feature_generator_path(base):
    return os.path.join(base, "FFN/feature_generator")


def _input_feature_dim_path(base):
    return os.path.join(base, "FFN/input_feature_dim")

def collate_fn(x):
    Z = []
    labels = []

    for z, label in x:
        Z.append(z)
        labels.append(label)

    Z = torch.tensor(Z)
    labels = torch.tensor(labels)
    return Z, labels

class FFNDataset(Dataset):
    def __init__(self, Z, latency):
        self.Z = Z
        self.latency = latency

    def __len__(self):
        return len(self.Z)

    def __getitem__(self, idx):
        return self.Z[idx], self.latency[idx]

def split_dataset(Z, latency, train_ratio=0.8):
    assert len(latency) == len(Z)

    total_size = len(Z)
    indices = list(range(total_size))
    np.random.shuffle(indices)

    # Determine the split sizes
    train_size = int(train_ratio * total_size)

    # Split the indices
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # Create sub-datasets
    train_dataset = FFNDataset([Z[i] for i in train_indices],
                               [latency[i] for i in train_indices])

    test_dataset = FFNDataset([Z[i] for i in test_indices],
                               [latency[i] for i in test_indices])

    return train_dataset, test_dataset

class OneHotNN(nn.Module):
    def __init__(self, num_classes):
        super(OneHotNN, self).__init__()
        self.num_classes = num_classes

    def forward(self, x):
        # Batch x 1
        one_hot = torch.zeros(x.size(0), self.num_classes, device=x.device)
        one_hot.scatter_(1, x.long(), 1)
        return one_hot

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
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
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
        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        return x

class FFN():
    def __init__(self, template_id, preprocessing_infos, device) -> None:
        super(FFN, self).__init__()
        self._input_feature_dim = None
        self._model_parallel = None
        self._template_id = template_id
        self.preprocessing_infos = preprocessing_infos
        self.device = device

    def load(self, path, fist_layer=0):
        with open(_input_feature_dim_path(path), "rb") as f:
            self._input_feature_dim = joblib.load(f)

        self.parameter_net = ParameterEmbeddingNet(self._template_id, self.preprocessing_infos).to(self.device)
        if fist_layer:
            state_dicts = torch.load(_fnn_path_first_layer(path, self._template_id),
                                     map_location=torch.device(self.device))
            self.parameter_net.embed_layers.load_state_dict(state_dicts['embed_layer'])
            self.parameter_net.fc1.load_state_dict(state_dicts['fc1'])
        state_dicts = torch.load(_fnn_path(path), map_location=torch.device(self.device))
        self.parameter_net.fc2.load_state_dict(state_dicts['fc2'])
        self.parameter_net.fc3.load_state_dict(state_dicts['fc3'])
        self.parameter_net.fc3.load_state_dict(state_dicts['fc4'])
        self.parameter_net.eval()


    def save(self, path):
        os.makedirs(f'{path}/FFN/', exist_ok=True)

        torch.save({
            'embed_layer': self.parameter_net.embed_layers.state_dict(),
            'fc1': self.parameter_net.fc1.state_dict()
        }, _fnn_path_first_layer(path, self._template_id))
        torch.save({
            'fc2': self.parameter_net.fc2.state_dict(),
            'fc3': self.parameter_net.fc3.state_dict(),
            'fc4': self.parameter_net.fc4.state_dict()
        }, _fnn_path(path))

        with open(_input_feature_dim_path(path), "wb") as f:
            joblib.dump(self._input_feature_dim, f)

    def fit(self, Z, latency, pre_training=False, batch_size=16, epochs=50):
        assert len(Z) == len(latency)

        if not pre_training:
            self.parameter_net = ParameterEmbeddingNet(self._template_id, self.preprocessing_infos).to(self.device)

        train_dataset, test_dataset = split_dataset(Z, latency)
        train_dataloader = DataLoader(train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                collate_fn=collate_fn)
        test_dataloader = DataLoader(test_dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      collate_fn=collate_fn)
        parameter_optimizer = torch.optim.Adam(self.parameter_net.parameters())

        bce_loss_fn = torch.nn.BCELoss()

        start_time = time()
        for epoch in range(epochs):
            loss_accum = 0
            for z, label in train_dataloader:
                z = z.to(self.device)
                label = label.to(self.device)

                z_pred = self.parameter_net(z)

                loss = bce_loss_fn(z_pred.squeeze(1), label)
                loss_accum += loss.item()

                parameter_optimizer.zero_grad()
                loss.backward()
                parameter_optimizer.step()

            loss_accum /= len(train_dataset)
            print("Epoch", epoch, "training loss:", loss_accum)

        end_time = time()
        print(f"Total training time: {(end_time-start_time)/1000}s")

        loss = self.evaluate(test_dataset, test_dataloader)
        print("test loss:", loss)

    def evaluate(self, dataset, dataloader):
        bce_loss_fn = torch.nn.BCELoss()

        self.parameter_net.eval()

        loss_accum = 0

        with torch.no_grad():
            for z, label in dataloader:
                z = z.to(self.device)
                label = label.to(self.device)

                z_pred = self.parameter_net(z)

                loss = bce_loss_fn(z_pred.squeeze(1), label)
                loss_accum += loss.item()

        # Compute average loss
        avg_loss = loss_accum / len(dataset)

        self.parameter_net.train()

        return avg_loss