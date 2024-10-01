import json
import os

from baseline.logPQO.model import XGBoostRegressorModel
from feature import FeatureGenerator


def _param_path(base):
    return os.path.join(base, "parameter.json")


def _cost_path(base):
    return os.path.join(base, "cost_matrix_baseline1.json")


def _meta_path(base):
    return os.path.join(base, "meta_data.json")

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



def load_training_data(training_data_file, template_id, k):
    path = os.path.join(training_data_file, template_id)
    X, Y = [], []

    with open(_param_path(path), 'r') as f:
        param = json.load(f)

    with open(_meta_path(path), 'r') as f:
        meta = json.load(f)

    with open(_cost_path(path), 'r') as f:
        cost = json.load(f)

    with open(os.path.join(path, f"selected_plan_baseline1_{k}.json"), 'r') as f:
        plan = json.load(f)

    for plan_key, _ in plan.items():
        x, y = [], []
        for param_key, _ in cost.items():
            x.append(param[param_key])
            y.append(cost[param_key][plan_key])
            if len(x) >= 20000:
                break
        X.append(x)
        Y.append(y)

    params, preprocess_info = get_param_info(meta)

    return X, Y, params, preprocess_info


def train(training_data, model_path, device):
    all_folders = []
    for subdir, _, files in os.walk(training_data):
        if 'a' in os.path.basename(subdir):
            all_folders.append(os.path.basename(subdir))

    print(all_folders)

    feature_generator = FeatureGenerator()

    for template_id in all_folders:
        for k in [10, 20, 30, 40, 50]:

            X, Y, params, preprocess_info = load_training_data(training_data, template_id, k)

            for i in range(k):
                regressor = XGBoostRegressorModel(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                                                  max_depth=5, alpha=10, n_estimators=10)
                x = feature_generator.transform_z(X[i], params, preprocess_info)
                regressor.train(x, Y[i])

                regressor.evaluate()

                # Save the model
                regressor.save_model(f'{model_path}/{template_id}/{k}/baseline_{i}.pkl')
