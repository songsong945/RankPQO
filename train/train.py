import argparse
import json

from feature import FeatureGenerator
from model import RankPQOModel


def get_training_pair(candidates):
    assert len(candidates) >= 2
    X1, X2 = [], []

    i = 0
    while i < len(candidates) - 1:
        s1 = candidates[i]
        j = i + 1
        while j < len(candidates):
            s2 = candidates[j]
            X1.append(s1)
            X2.append(s2)
            j += 1
        i += 1
    return X1, X2


def _load_pairwise_plans(path):
    X1, X2 = [], []
    with open(path, 'r') as f:
        for line in f.readlines():
            arr = line.split("#####")
            x1, x2 = get_training_pair(arr)
            X1 += x1
            X2 += x2
    return X1, X2


def training_pairwise(training_data_file, model_path, template_id, pre_train):
    X1, X2, Z = _load_pairwise_plans(training_data_file)

    tuning_model = model_path is not None
    rank_PQO_model = None
    if tuning_model:
        rank_PQO_model = RankPQOModel(None, template_id)
        rank_PQO_model.load(model_path)
        feature_generator = rank_PQO_model._feature_generator
    else:
        feature_generator = FeatureGenerator()
        feature_generator.fit(X1 + X2)

    Y1, Y2 = None, None
    if pre_train:
        Y1 = [json.loads(c)[0]['Plan']['Total Cost'] for c in X1]
        Y2 = [json.loads(c)[0]['Plan']['Total Cost'] for c in X2]
        X1, _ = feature_generator.transform(X1)
        X2, _ = feature_generator.transform(X2)
        Z = feature_generator.transform_z(Z)
    else:
        X1, Y1 = feature_generator.transform(X1)
        X2, Y2 = feature_generator.transform(X2)
        Z = feature_generator.transform_z(Z)
    print("Training data set size = " + str(len(X1)))

    if not tuning_model:
        assert rank_PQO_model is None
        rank_PQO_model = RankPQOModel(feature_generator, template_id)
    rank_PQO_model.fit(X1, X2, Y1, Y2, Z, tuning_model)

    print("saving model...")
    rank_PQO_model.save(model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Model training helper")
    parser.add_argument("--training_data", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--template_id", type=int)
    parser.add_argument("--pre_trained", type=int)

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

    training_pairwise(training_data, model_path, template_id, pre_trained)
