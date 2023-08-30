import hashlib
import json
import os


def get_structural_representation(plan, depth=0):
    node_type = plan['Node Type']

    if 'Plans' not in plan:
        # 叶子节点
        table_name = plan.get('Relation Name', 'unknown')
        return [(node_type, table_name, depth)]
    else:
        # 内部节点
        sub_structure = [item for subplan in plan['Plans'] for item in
                         get_structural_representation(subplan, depth + 1)]
        return [(node_type, depth)] + sub_structure


def compute_hash(representation):
    m = hashlib.md5()
    m.update(str(representation).encode('utf-8'))
    return m.hexdigest()


def deduplicate_plans(plan_file_path):
    with open(plan_file_path, 'r') as file:
        plans = json.load(file)

    unique_plans = {}
    seen_hashes = set()

    for plan_name, plan in plans.items():
        representation = get_structural_representation(plan['Plan'])
        hash_val = compute_hash(representation)
        if hash_val not in seen_hashes:
            seen_hashes.add(hash_val)
            unique_plans[plan_name] = plan

    return unique_plans


def process_all_plans(data_directory):
    for subdir, _, _ in os.walk(data_directory):
        plan_file_path = os.path.join(subdir, "all_plans.json")

        if os.path.isfile(plan_file_path):
            print(f"Processing {plan_file_path}...")
            unique_plans = deduplicate_plans(plan_file_path)

            with open(os.path.join(subdir, "plan.json"), 'w') as out_file:
                json.dump(unique_plans, out_file, indent=4)


if __name__ == "__main__":
    meta_data_path = '../training_data/JOB/'
    process_all_plans(meta_data_path)
