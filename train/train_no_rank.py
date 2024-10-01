import argparse
import json
import os
import random
import sys
import time


sys.path.append('..')

from model.feature import FeatureGenerator
from model.model_no_rank import RankPQOModel

k1 = [7, 9, 11, 13, 15]
k2 = [15, 20, 25, 30, 32]
k3 = [7, 9, 11, 13, 15, 16, 17, 18, 19, 20]  # 1k - 10k

random.seed(42)

def _param_path(base):
    return os.path.join(base, "parameter_new.json")


# def _cost_path(base):
#     return os.path.join(base, "cost_matrix_2.json")

def _cost_path(base):
    return os.path.join(base, "latency_matrix_new.json")


def _meta_path(base):
    return os.path.join(base, "meta_data.json")


def _plan_path(base):
    return os.path.join(base, "all_plans_by_hybrid_new.json")

# def _plan_path(base):
#     return os.path.join(base, "hybrid_plans.json")


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

            if "distinct_values" in data:
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
        # if i > k2[4]:
        #     break
    return X1, X2, Y1, Y2

def get_training_pair_k(candidate_plan, plan, param_key, cost, k):
    assert len(candidate_plan) >= 2
    X1, X2, Y1, Y2 = [], [], [], []

    for i in range(k):
        sampled_indices = random.sample(range(len(candidate_plan)), 2)
        s1 = candidate_plan[sampled_indices[0]]
        s2 = candidate_plan[sampled_indices[1]]
        X1.append(plan[s1])
        Y1.append(cost[param_key][s1])
        X2.append(plan[s2])
        Y2.append(cost[param_key][s2])

    return X1, X2, Y1, Y2

def load_training_data_k(training_data_file, template_id, k):
    path = os.path.join(training_data_file, template_id)
    Z, X1, X2, Y1, Y2, params, preprocess_info = [], [], [], [], [], [], []
    random.seed(42)

    with open(_param_path(path), 'r') as f:
        param = json.load(f)

    with open(_cost_path(path), 'r') as f:
        cost = json.load(f)

    with open(_meta_path(path), 'r') as f:
        meta = json.load(f)

    with open(_plan_path(path), 'r') as f:
        plan = json.load(f)

    param_keys = list(cost.keys())
    param_num = len(param_keys)

    for param_key in param_keys:
        param_values = param[param_key]
        candidate_plan = list(cost[param_key].keys())
        x1, x2, y1, y2 = get_training_pair_k(candidate_plan, plan, param_key, cost, int(k/param_num))
        Z += [list(param_values) for _ in range(len(x1))]
        X1 += x1
        X2 += x2
        Y1 += y1
        Y2 += y2

    params, preprocess_info = get_param_info(meta)
    return Z, X1, X2, Y1, Y2, params, preprocess_info

def load_training_data(training_data_file, template_id):
    path = os.path.join(training_data_file, template_id)
    Z, X1, X2, Y1, Y2, params, preprocess_info = [], [], [], [], [], [], []
    random.seed(42)

    with open(_param_path(path), 'r') as f:
        param = json.load(f)

    with open(_cost_path(path), 'r') as f:
        cost = json.load(f)

    with open(_meta_path(path), 'r') as f:
        meta = json.load(f)

    with open(_plan_path(path), 'r') as f:
        plan = json.load(f)

    param_keys = random.sample(cost.keys(), min(50, len(cost.keys())))

    for param_key in param_keys:
        param_values = param[param_key]
        candidate_plan = random.sample(list(cost[param_key].keys()), min(50, len(cost[param_key].keys())))
        x1, x2, y1, y2 = get_training_pair(candidate_plan, plan, param_key, cost)
        Z += [list(param_values) for _ in range(len(x1))]
        X1 += x1
        X2 += x2
        Y1 += y1
        Y2 += y2
        # if len(X1) >= k1[4]:
        #     size = k1[4]
        #     Z = Z[:size]
        #     X1 = X1[:size]
        #     X2 = X2[:size]
        #     Y1 = Y1[:size]
        #     Y2 = Y2[:size]
        #     break

    params, preprocess_info = get_param_info(meta)
    return Z, X1, X2, Y1, Y2, params, preprocess_info


def training_pairwise(training_data_file, model_path, template_id, device, pre_trained=0, first_layer=0):
    Z, X1, X2, Y1, Y2, params, preprocess_info = load_training_data(training_data_file, template_id)

    rank_PQO_model = None
    if pre_trained:
        rank_PQO_model = RankPQOModel(None, template_id, preprocess_info, device=device)
        rank_PQO_model.load(model_path, first_layer)
        feature_generator = rank_PQO_model._feature_generator
    else:
        feature_generator = FeatureGenerator()
        feature_generator.fit_pred_model(X1 + X2)

    X1 = feature_generator.transform(X1)
    X2 = feature_generator.transform(X2)
    Z = feature_generator.transform_z(Z, params, preprocess_info)
    print("Training data set size = " + str(len(X1)))

    if not pre_trained:
        assert rank_PQO_model is None
        rank_PQO_model = RankPQOModel(feature_generator, template_id, preprocess_info, device=device)
    rank_PQO_model.fit_with_test(X1, X2, Y1, Y2, Z, pre_trained)

    print("saving model...")
    rank_PQO_model.save(model_path)


def training_no_share(training_data, model_path, device):
    # all_folders = [f for f in os.listdir(training_data) if os.path.isdir(os.path.join(training_data, f))]
    # all_folders = [
    #     "30c", "16c", "21c", "9a", "13b", "13c", "15a", "4a", "23b", "33a", "27b", "2b", "31a"
    # ]
    all_folders = []
    for subdir, _, files in os.walk(training_data):
        if ("meta_data.json" in files and "hybrid_plans.json" in files
                and "parameters.json" in files and "cost_matrix_2.json" in files):
            all_folders.append(os.path.basename(subdir))

    print(all_folders)
    io_time = 0
    training_time = 0

    X_all = None
    for folder in all_folders:
        Z, X1, X2, Y1, Y2, params, preprocess_info = load_training_data_k(training_data, folder, 3000)
        if X_all is None:
            X_all = (X1 + X2)
        else:
            X_all += (X1 + X2)

    for folder in all_folders:
        print(f"Training for folder: {folder}")
        model_path_file = os.path.join(model_path, folder)

        t1 = time.time()
        Z, X1, X2, Y1, Y2, params, preprocess_info = load_training_data_k(training_data, folder, 3000)
        t2 = time.time()

        feature_generator = FeatureGenerator(False, True)
        feature_generator.fit_pred_model(X_all)

        X1 = feature_generator.transform(X1)
        X2 = feature_generator.transform(X2)
        Z = feature_generator.transform_z(Z, params, preprocess_info)
        print("Training data set size = " + str(len(X1)))

        rank_PQO_model = RankPQOModel(feature_generator, folder, preprocess_info, device=device)

        rank_PQO_model.fit_with_test(X1, X2, Y1, Y2, Z, pre_trained, 16, 50)

        rank_PQO_model.save(model_path_file)

        print(f"Finished training for folder: {folder}")

        t3 = time.time()

        io_time += (t2 - t1)
        training_time += (t3 - t2)

    for folder in all_folders:
        template_id = folder
        model_path_file = os.path.join(model_path, folder)

        Z, X1, X2, Y1, Y2, params, preprocess_info = load_training_data_k(training_data, template_id, 3000)
        rank_PQO_model = RankPQOModel(None, template_id, preprocess_info, device=device)
        rank_PQO_model.load(model_path_file, fist_layer=1)
        feature_generator = rank_PQO_model._feature_generator
        X1 = feature_generator.transform(X1)
        X2 = feature_generator.transform(X2)
        Z = feature_generator.transform_z(Z, params, preprocess_info)
        print(f"Test results for {folder}")
        rank_PQO_model.test(X1, X2, Y1, Y2, Z, 16)

    print(f'io time: {io_time}')
    print(f'train time: {training_time}')


def training_baseline(training_data, model_path, device):
    # all_folders = [f for f in os.listdir(training_data) if os.path.isdir(os.path.join(training_data, f))]
    all_folders = []
    for subdir, _, files in os.walk(training_data):
        if ("meta_data.json" in files and "hybrid_plans.json" in files
                and "parameters.json" in files and "cost_matrix_2.json" in files):
            all_folders.append(os.path.basename(subdir))

    print(all_folders)

    io_time = 0
    training_time = 0

    X_all = None
    for folder in all_folders:
        Z, X1, X2, Y1, Y2, params, preprocess_info = load_training_data(training_data, folder)
        if X_all is None:
            X_all = (X1 + X2)
        else:
            X_all += (X1 + X2)

    pre_trained = 0
    for folder in all_folders:
        print(f"Training for folder: {folder}")

        t1 = time.time()

        Z, X1, X2, Y1, Y2, params, preprocess_info = load_training_data(training_data, folder)

        t2 = time.time()

        if pre_trained:
            rank_PQO_model = RankPQOModel(None, folder, preprocess_info, device=device)
            rank_PQO_model.load(model_path)
            feature_generator = rank_PQO_model._feature_generator
        else:
            feature_generator = FeatureGenerator(False, True)
            feature_generator.fit_pred_model(X_all)

        X1 = feature_generator.transform(X1)
        X2 = feature_generator.transform(X2)
        Z = feature_generator.transform_z(Z, params, preprocess_info)
        print("Training data set size = " + str(len(X1)))

        if not pre_trained:
            rank_PQO_model = RankPQOModel(feature_generator, folder, preprocess_info, device=device)

        rank_PQO_model.fit_with_test(X1, X2, Y1, Y2, Z, pre_trained, 16, 50)

        rank_PQO_model.save(model_path)

        pre_trained = 1

        print(f"Finished training for folder: {folder}")

        t3 = time.time()

        io_time += (t2 - t1)
        training_time += (t3 - t2)

    for folder in all_folders:
        template_id = folder

        Z, X1, X2, Y1, Y2, params, preprocess_info = load_training_data(training_data, template_id)
        rank_PQO_model = RankPQOModel(None, template_id, preprocess_info, device=device)
        rank_PQO_model.load(model_path, fist_layer=1)
        feature_generator = rank_PQO_model._feature_generator
        X1 = feature_generator.transform(X1)
        X2 = feature_generator.transform(X2)
        Z = feature_generator.transform_z(Z, params, preprocess_info)
        print(f"Test results for {folder}")
        rank_PQO_model.test(X1, X2, Y1, Y2, Z, 16)

    print(f'io time: {io_time}')
    print(f'train time: {training_time}')


def alternating_training(training_data, model_path, device, epochs, epoch_step):
    # all_folders = ['query013', 'query018', 'query019', 'query025', 'query027', 'query040', 'query050', 'query072',
    #                'query084', 'query091', 'query099', 'query100', 'query101', 'query102']
    all_folders = []
    for subdir, _, files in os.walk(training_data):
        if ("meta_data.json" in files):
            all_folders.append(os.path.basename(subdir))

    print(all_folders)
    io_time = 0
    training_time = 0

    all_epochs = [epoch_step for _ in range(epochs // epoch_step)] + (
        [epochs % epoch_step] if epochs % epoch_step else [])

    pre_trained = 0
    epoch_counter = 0
    first_layer = 0
    X_all = None
    for folder in all_folders:
        Z, X1, X2, Y1, Y2, params, preprocess_info = load_training_data_k(training_data, folder, 3000)
        if X_all is None:
            X_all = (X1 + X2)
        else:
            X_all += (X1 + X2)
    for epoch in all_epochs:
        epoch_counter += epoch
        print(f"Training for total epochs: {epoch_counter}/{epochs}")

        for folder in all_folders:
            print(f"Training for folder: {folder}")

            t1 = time.time()

            Z, X1, X2, Y1, Y2, params, preprocess_info = load_training_data_k(training_data, folder, 3000)

            t2 = time.time()

            if pre_trained:
                rank_PQO_model = RankPQOModel(None, folder, preprocess_info, device=device)
                rank_PQO_model.load(model_path, first_layer)
                feature_generator = rank_PQO_model._feature_generator
            else:
                feature_generator = FeatureGenerator(False, True)
                # feature_generator = FeatureGenerator(True, False)
                feature_generator.fit_pred_model(X_all)

            X1 = feature_generator.transform(X1)
            X2 = feature_generator.transform(X2)
            Z = feature_generator.transform_z(Z, params, preprocess_info)

            print("Training data set size = " + str(len(X1)))

            if not pre_trained:
                rank_PQO_model = RankPQOModel(feature_generator, folder, preprocess_info, device=device)

            rank_PQO_model.fit_with_test(X1, X2, Y1, Y2, Z, pre_trained, 16, epoch)

            rank_PQO_model.save(model_path)

            pre_trained = 1

            print(f"Finished training for folder: {folder}")
            t3 = time.time()

            io_time += (t2 - t1)
            training_time += (t3 - t2)

        first_layer = 1
        random.shuffle(all_folders)
        print(all_folders)

    for folder in all_folders:
        template_id = folder

        Z, X1, X2, Y1, Y2, params, preprocess_info = load_training_data_k(training_data, folder, 3000)
        rank_PQO_model = RankPQOModel(None, template_id, preprocess_info, device=device)
        rank_PQO_model.load(model_path, fist_layer=1)
        feature_generator = rank_PQO_model._feature_generator
        X1 = feature_generator.transform(X1)
        X2 = feature_generator.transform(X2)
        Z = feature_generator.transform_z(Z, params, preprocess_info)
        print(f"Test results for {folder}")
        rank_PQO_model.test(X1, X2, Y1, Y2, Z, 16)

    print(f'io time: {io_time}')
    print(f'train time: {training_time}')


def test(training_data, model_path, device, is_share):
    # all_folders = [f for f in os.listdir(training_data) if os.path.isdir(os.path.join(training_data, f))]
    # all_folders = [
    #     "30c", "16c", "21c", "9a", "13b", "13c", "15a", "4a", "23b", "33a", "27b", "2b", "31a"
    # ]
    all_folders = []
    for subdir, _, files in os.walk(training_data):
        if ('a' in os.path.basename(subdir) and "meta_data.json" in files and "plan_by_join_order.json" in files
                and "parameter.json" in files and "latency_matrix.json" in files):
            all_folders.append(os.path.basename(subdir))
    for folder in all_folders:
        template_id = folder
        if is_share:
            model_path_file = model_path
        else:
            model_path_file = os.path.join(model_path, folder)

        Z, X1, X2, Y1, Y2, params, preprocess_info = load_training_data(training_data, template_id)
        rank_PQO_model = RankPQOModel(None, template_id, preprocess_info, device=device)
        rank_PQO_model.load(model_path_file, fist_layer=1)
        feature_generator = rank_PQO_model._feature_generator
        X1 = feature_generator.transform(X1)
        X2 = feature_generator.transform(X2)
        Z = feature_generator.transform_z(Z, params, preprocess_info)
        print(f"Test results for {folder}")
        rank_PQO_model.test(X1, X2, Y1, Y2, Z, 16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Model training helper")
    parser.add_argument("--function", type=str)
    parser.add_argument("--training_data", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--template_id", type=str)
    parser.add_argument("--pre_trained", type=int)
    parser.add_argument("--first_layer", type=int)
    parser.add_argument("--is_share", type=int)
    parser.add_argument("--device", type=str, default='cpu')

    args = parser.parse_args()

    function = None
    if args.function is not None:
        function = args.function
    print("function:", function)

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

    first_layer = False
    if args.first_layer is not None:
        first_layer = args.first_layer
    print("first_layer:", first_layer)

    is_share = False
    if args.is_share is not None:
        is_share = args.is_share
    print("is_share:", is_share)

    print("Device: ", args.device)

    if function == "training_baseline":
        training_baseline(training_data, model_path, args.device)
    elif function == "alternating_training":
        alternating_training(training_data, model_path, args.device, 10, 1)
    elif function == "training_no_share":
        training_no_share(training_data, model_path, args.device)
    elif function == "test":
        test(training_data, model_path, args.device, is_share)
    else:
        training_pairwise(training_data, model_path, template_id, args.device, pre_trained, first_layer)
