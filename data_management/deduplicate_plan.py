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


def generate_hint_from_plan(plan):
    node = plan['Plan']
    hints = []

    def traverse_node(node):
        node_type = node['Node Type']

        # PG uses the former & the extension expects the latter.
        node_type = node_type.replace(' ', '')
        node_type = node_type.replace('Nested Loop', 'NestLoop')

        if 'Relation Name' in node:  # If it's a scan operation
            relation = node['Relation Name']
            hint = node_type + '(' + relation + ')'
            hints.append(hint)
            return [relation], relation
        else:
            rels = []  # Flattened
            leading = []  # Hierarchical
            if 'Plans' in node:
                for child in node['Plans']:
                    a, b = traverse_node(child)
                    rels.extend(a)
                    leading.append(b)
            join_hint = node_type + '(' + ' '.join(rels) + ')'
            hints.append(join_hint)
            return rels, leading

    _, leading_hierarchy = traverse_node(node)

    leading = 'Leading(' + str(leading_hierarchy) \
        .replace('\'', '') \
        .replace('[', '(') \
        .replace(']', ')') \
        .replace(',', '') + ')'

    hints.append(leading)

    query_hint = '\n '.join(hints)
    return query_hint


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
        # representation = get_structural_representation(plan['Plan'])
        representation = generate_hint_from_plan(plan)
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

            with open(os.path.join(subdir, "plan2.json"), 'w') as out_file:
                json.dump(unique_plans, out_file, indent=4)


if __name__ == "__main__":
    meta_data_path = '../training_data/JOB/'
    process_all_plans(meta_data_path)
