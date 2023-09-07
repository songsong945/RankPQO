import json
import os


def update_meta(meta_data_path):
    # 读取meta_data
    with open(meta_data_path, 'r') as file:
        meta_data = json.load(file)

    for predicate in meta_data["predicates"]:
        if predicate["data_type"] == "int":
            predicate["max_len"] = predicate["max"] - predicate["min"] + 1

    # 更新meta_data.json文件
    with open(meta_data_path, 'w') as file:
        json.dump(meta_data, file, indent=4)


def process_all_meta_data(data_directory):
    for subdir, _, _ in os.walk(data_directory):
        meta_data_path = os.path.join(subdir, "meta_data.json")

        # 检查meta_data.json是否在子目录中
        if os.path.isfile(meta_data_path):
            print(f"Processing: {meta_data_path}")
            update_meta(meta_data_path)


if __name__ == "__main__":
    meta_data_path = '../training_data/JOB/'
    process_all_meta_data(meta_data_path)
