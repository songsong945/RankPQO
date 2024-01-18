import json
import os


def update_meta(meta_data_path, parameter_path):
    # 读取meta_data
    with open(meta_data_path, 'r') as f:
        meta_data = json.load(f)
    with open(parameter_path, 'r') as f:
        parameters = json.load(f)

    for predicate in meta_data['predicates']:
        if predicate['data_type'] == 'text' or predicate['data_type'] == 'int':
            # 获取谓词的别名和列名

            distinct_values_in_params = list(set(
                param[meta_data['predicates'].index(predicate)] for param in parameters.values()))
            predicate['distinct_values'] = distinct_values_in_params
            predicate['max_len'] = len(distinct_values_in_params) + 1
            predicate['preprocess_type'] = "embedding"

    # 写回文件
    with open(meta_data_path, 'w') as f:
        json.dump(meta_data, f, indent=4)


def process_all_meta_data(data_directory):
    for subdir, _, _ in os.walk(data_directory):
        meta_data_path = os.path.join(subdir, "meta_data.json")
        parameter_path = os.path.join(subdir, "parameter_new.json")

        # 检查meta_data.json是否在子目录中
        if os.path.isfile(meta_data_path) and 'a' in os.path.basename(subdir):
            print(f"Processing: {meta_data_path}")
            update_meta(meta_data_path, parameter_path)


if __name__ == "__main__":
    meta_data_path = '../training_data/JOB/'
    process_all_meta_data(meta_data_path)