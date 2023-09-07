import os
import time

import psycopg2
import json

import configure
from multiprocessing import Pool


def connect_to_pg():
    connection = psycopg2.connect(
        dbname=configure.dbname,
        user=configure.user,
        password=configure.password,
        host=configure.host,
        port=configure.port
    )
    return connection


def generate_hint_from_plan(plan):
    node = plan['Plan']
    hints = []

    def traverse_node(node):
        node_type = node['Node Type']
        rels = []  # Flattened list of relation names/aliases
        leading = []  # Hierarchical structure for LEADING hint

        # PG uses the former & the extension expects the latter.
        node_type = node_type.replace(' ', '')
        node_type = node_type.replace('NestedLoop', 'NestLoop')

        if 'Relation Name' in node:  # If it's a scan operation
            relation = node.get('Alias', node['Relation Name'])  # Prefer alias if exists
            if node_type in ['IndexScan', 'SeqScan']:
                hint = node_type + '(' + relation + ')'
                hints.append(hint)
            return [relation], relation
        else:
            if 'Plans' in node:
                for child in node['Plans']:
                    a, b = traverse_node(child)
                    rels.extend(a)
                    if b:  # Only add if it's not None
                        leading.append(b)
            if node_type in ['HashJoin', 'MergeJoin', 'NestLoop']:
                join_hint = node_type + '(' + ' '.join(rels) + ')'
                hints.append(join_hint)
            return rels, leading

    _, leading_hierarchy = traverse_node(node)

    def pair_hierarchy(hierarchy):
        if isinstance(hierarchy, str):
            return hierarchy
        elif len(hierarchy) == 1:
            return pair_hierarchy(hierarchy[0])
        else:
            hierarchy = [pair_hierarchy(item) for item in hierarchy]
        return hierarchy

    leading_hierarchy = pair_hierarchy(leading_hierarchy)

    leading_hierarchy = str(leading_hierarchy).replace('\'', '') \
        .replace(',', '')

    leading = 'Leading(' + leading_hierarchy + ')'

    hints.append(leading)

    query_hint = '\n '.join(hints)
    return query_hint


def fetch_plan_cost(connection, query_with_hint, parameters):
    cursor = connection.cursor()
    cursor.execute("SET max_parallel_workers_per_gather TO 0;")

    query_with_hint = cursor.mogrify(query_with_hint, parameters).decode()
    cursor.execute(f"EXPLAIN (FORMAT JSON) {query_with_hint}")
    plan = cursor.fetchone()
    cursor.close()
    cost = plan[0][0]['Plan']['Total Cost']
    return cost


def fetch_actual_latency(connection, query_with_hint, parameters):
    cursor = connection.cursor()
    cursor.execute("SET max_parallel_workers_per_gather TO 0;")

    query_with_hint = cursor.mogrify(query_with_hint, parameters).decode()

    start_time = time.time()
    cursor.execute(query_with_hint)
    end_time = time.time()

    cursor.close()
    latency = end_time - start_time
    return latency


def evaluate_plans_for_parameters(connection, meta_data, plans, parameters_data):
    template = meta_data["template"]
    results = {}

    for param_key, param_values in parameters_data.items():
        results[param_key] = {}
        # print(f"    Processing {param_key}...")
        for plan_key, plan in plans.items():
            plan_hint = generate_hint_from_plan(plan)
            query_with_hint = f"/*+ {plan_hint} */ " + template

            query_with_hint = query_with_hint.format(*param_values)

            # cost = fetch_plan_cost(connection, query_with_hint, param_values)
            cost = fetch_actual_latency(connection, query_with_hint, param_values)
            results[param_key][plan_key] = cost

    return results


def evaluate_all(data_directory):
    connection = connect_to_pg()

    for subdir, _, files in os.walk(data_directory):
        if "meta_data.json" in files and "plan.json" in files and "parameter.json" in files:
            with open(os.path.join(subdir, "meta_data.json"), 'r') as f_meta:
                meta_data = json.load(f_meta)

            with open(os.path.join(subdir, "plan.json"), 'r') as f_plans:
                plans = json.load(f_plans)

            with open(os.path.join(subdir, "parameter.json"), 'r') as f_params:
                parameters = json.load(f_params)

            print(f"Processing {subdir}...")

            costs = evaluate_plans_for_parameters(connection, meta_data, plans, parameters)

            with open(os.path.join(subdir, "latency_matrix.json"), 'w') as f_costs:
                json.dump(costs, f_costs, indent=4)

    connection.close()


def evaluate_directory(subdir):
    connection = connect_to_pg()

    with open(os.path.join(subdir, "meta_data.json"), 'r') as f_meta:
        meta_data = json.load(f_meta)

    with open(os.path.join(subdir, "plan.json"), 'r') as f_plans:
        plans = json.load(f_plans)

    with open(os.path.join(subdir, "parameter.json"), 'r') as f_params:
        parameters = json.load(f_params)

    print(f"Processing {subdir}...")

    costs = evaluate_plans_for_parameters(connection, meta_data, plans, parameters)

    with open(os.path.join(subdir, "latency_matrix.json"), 'w') as f_costs:
        json.dump(costs, f_costs, indent=4)

    connection.close()


def evaluate_all_mutil_process(data_directory):
    directories_to_process = []

    for subdir, _, files in os.walk(data_directory):
        if "meta_data.json" in files and "plan.json" in files and "parameter.json" in files:
            directories_to_process.append(subdir)

    # Create a Pool with the desired number of processes
    with Pool(processes=8) as pool:  # for example, 4 processes
        pool.map(evaluate_directory, directories_to_process)


if __name__ == "__main__":
    meta_data_path = '../training_data/JOB/'
    evaluate_all_mutil_process(meta_data_path)
