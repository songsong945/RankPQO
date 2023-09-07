import argparse
import json
import os
import sys

sys.path.append('..')

from model.feature import FeatureGenerator
from model.model import RankPQOModel


def _param_path(base):
    return os.path.join(base, "parameter.json")


def _cost_path(base):
    return os.path.join(base, "cost_matrix.json")


def _meta_path(base):
    return os.path.join(base, "meta_data.json")


def _plan_path(base):
    return os.path.join(base, "plan.json")


def get_param_info(meta):
    params, preprocess_info = [], []

    for data in meta["predicates"]:
        param_data = {}
        preprocess_info_data = {}

        if data["data_type"] in ["int", "float", "text"]:
            param_data["data_type"] = data["data_type"]

            if data["data_type"] == "int" and "min" in data and "max" in data:
                param_data["min"] = data["min"]
                param_data["max"] = data["max"]

            if data["data_type"] == "text" and "distinct_values" in data:
                param_data["distinct_values"] = data["distinct_values"]

            params.append(param_data)

        if data["preprocess_type"] in ["one_hot", "std_normalization", "embedding"]:
            preprocess_info_data["type"] = data["preprocess_type"]

            if data["preprocess_type"] == "one_hot" and "max_len" in data:
                preprocess_info_data["max_len"] = data["max_len"]

            if data["preprocess_type"] == "std_normalization" and "mean" in data and "variance" in data:
                preprocess_info_data["mean"] = data["mean"]
                preprocess_info_data["variance"] = data["variance"]

            if data["preprocess_type"] == "embedding" and "max_len" in data:
                preprocess_info_data["max_len"] = data["max_len"]

            preprocess_info.append(preprocess_info_data)

    return params, preprocess_info


def get_training_pair(candidate_plan, plan, param_key, cost):
    assert len(candidate_plan) >= 2
    X1, X2, Y1, Y2 = [], [], [], []

    i = 0
    while i < len(candidate_plan) - 1:
        s1 = candidate_plan[i]
        j = i + 1
        while j < len(candidate_plan):
            s2 = candidate_plan[j]
            X1.append(plan[s1])
            Y1.append(cost[param_key][s1])
            X2.append(plan[s2])
            Y2.append(cost[param_key][s2])
            j += 1
        i += 1
    return X1, X2, Y1, Y2


def load_training_data(training_data_file, template_id):
    path = os.path.join(training_data_file, template_id)
    Z, X1, X2, Y1, Y2, params, preprocess_info = [], [], [], [], [], [], []

    with open(_param_path(path), 'r') as f:
        param = json.load(f)

    with open(_cost_path(path), 'r') as f:
        cost = json.load(f)

    with open(_meta_path(path), 'r') as f:
        meta = json.load(f)

    with open(_plan_path(path), 'r') as f:
        plan = json.load(f)

    for param_key, param_values in param.items():
        candidate_plan = list(cost[param_key].keys())
        x1, x2, y1, y2 = get_training_pair(candidate_plan, plan, param_key, cost)
        Z += [list(param_values) for _ in range(len(x1))]
        X1 += x1
        X2 += x2
        Y1 += y1
        Y2 += y2

    params, preprocess_info = get_param_info(meta)

    return Z, X1, X2, Y1, Y2, params, preprocess_info


def training_pairwise(training_data_file, model_path, template_id, device, pre_trained=0):
    Z, X1, X2, Y1, Y2, params, preprocess_info = load_training_data(training_data_file, template_id)

    tuning_model = model_path is not None
    rank_PQO_model = None
    if pre_trained:
        rank_PQO_model = RankPQOModel(None, template_id, None)
        rank_PQO_model.load(model_path)
        feature_generator = rank_PQO_model._feature_generator
    else:
        feature_generator = FeatureGenerator()
        feature_generator.fit(X1 + X2)

    X1 = feature_generator.transform(X1)
    X2 = feature_generator.transform(X2)
    Z = feature_generator.transform_z(Z, params, preprocess_info)
    print("Training data set size = " + str(len(X1)))

    if not pre_trained:
        assert rank_PQO_model is None
        rank_PQO_model = RankPQOModel(feature_generator, template_id, preprocess_info, device=device)
    rank_PQO_model.fit(X1, X2, Y1, Y2, Z, pre_trained)

    print("saving model...")
    rank_PQO_model.save(model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Model training helper")
    parser.add_argument("--training_data", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--template_id", type=str)
    parser.add_argument("--pre_trained", type=int)
    parser.add_argument("--device", type=str, default='cpu')

    args = parser.parse_args()

    training_data = None
    if args.training_data is not None:
        training_data = args.training_data
    print("training_data:", training_data)

    model_path = None
    if args.model_path is not None:
        model_path = args.model_path
    print("model_path:", model_path)

    template_id = False
    if args.template_id is not None:
        template_id = args.template_id
    print("template_id:", template_id)

    pre_trained = False
    if args.pre_trained is not None:
        pre_trained = args.pre_trained
    print("pre_trained:", pre_trained)

    print("Device: ", args.device)

    training_pairwise(training_data, model_path, template_id, args.device, pre_trained)
