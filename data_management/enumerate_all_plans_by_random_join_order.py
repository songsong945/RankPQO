import random
import time

import psycopg2
import json
import os
import configure
import itertools

import numpy as np
from multiprocessing import Pool


def extract_tables_from_node(node):
    """
    Recursively extract table aliases from a plan node.
    """
    tables = []

    if "Relation Name" in node:
        tables.append(node["Alias"])

    if "Plans" in node:
        for child in node["Plans"]:
            tables.extend(extract_tables_from_node(child))

    return tables


def generate_join_order_hints(plan, k):
    if isinstance(plan, str):
        plan = json.loads(plan)

    tables = extract_tables_from_node(plan["Plan"])

    table_permutations = itertools.permutations(tables)

    print("----")

    # Sample k permutations from the generator
    sampled_permutations = list(itertools.islice(table_permutations, k))

    hints = []
    for perm in sampled_permutations:
        hint = "/*+ Leading(" + " ".join(perm) + ") */"
        hints.append(hint)

    return hints


def connect_to_pg():
    connection = psycopg2.connect(
        dbname=configure.dbname,
        user=configure.user,
        password=configure.password,
        host=configure.host,
        port=configure.port
    )
    return connection


def fetch_execution_plan(connection, template, parameters):
    cursor = connection.cursor()

    cursor.execute("SET max_parallel_workers_per_gather TO 0;")

    query = cursor.mogrify(template, parameters).decode()
    cursor.execute(f"EXPLAIN (FORMAT JSON) {query}")
    plan = cursor.fetchone()
    cursor.close()
    return plan[0][0]


def generate_plans_for_query(meta_data_path, parameter_path):
    with open(meta_data_path, 'r') as f:
        meta_data = json.load(f)
    with open(parameter_path, 'r') as f:
        parameters_list = json.load(f)

    connection = connect_to_pg()
    plans = {}

    # table_aliases = [predicate["alias"] for predicate in meta_data["predicates"]]

    idx = 0
    for params in enumerate(parameters_list.values()):

        plan = fetch_execution_plan(connection, meta_data['template'], params[1])
        idx += 1
        plans[f"plan {idx}"] = plan

        # for alias in table_aliases:
        #     hints = generate_hints_from_plan(plan, alias)
        hints = generate_join_order_hints(plan, 50)
        for hint in hints:
            idx += 1
            modified_plan_with_hint = fetch_execution_plan(connection, hint + " " + meta_data['template'],
                                                           params[1])
            plans[f"modified {idx}"] = modified_plan_with_hint

    connection.close()
    return plans


def generate_plans_for_query_by_distinct_plan(meta_data_path, parameter_path, plan_json_path):
    with open(meta_data_path, 'r') as f:
        meta_data = json.load(f)
    with open(parameter_path, 'r') as f:
        parameters_list = json.load(f)
    with open(plan_json_path, 'r') as f:
        plan_data = json.load(f)

    connection = connect_to_pg()
    plans = {}

    # 从plan.json中的plan IDs获取实际的ID
    plan_ids = [int(key.split(" ")[1]) for key in plan_data.keys()]

    for plan_id in plan_ids:
        plans[f"plan {plan_id}"] = plan_data[f"plan {plan_id}"]

    sampled_plan_ids = plan_ids[:20]
    idx = max(plan_ids)

    for plan_id in sampled_plan_ids:
        params = parameters_list[f"parameter {plan_id}"]
        plan = plan_data[f"plan {plan_id}"]

        hints = generate_join_order_hints(plan, 200)
        for hint in hints:
            idx += 1
            modified_plan_with_hint = fetch_execution_plan(connection, hint + " " + meta_data['template'], params)
            plans[f"plan {idx}"] = modified_plan_with_hint

    connection.close()
    return plans


def save_execution_plans_for_all(data_directory):
    for subdir, _, _ in os.walk(data_directory):
        meta_data_path = os.path.join(subdir, "meta_data.json")
        parameter_path = os.path.join(subdir, "parameter.json")

        if os.path.isfile(meta_data_path) and os.path.isfile(parameter_path):
            print(f"Processing: {meta_data_path}")
            plans = generate_plans_for_query_by_distinct_plan(meta_data_path, parameter_path)

            with open(os.path.join(subdir, "all_plans_by_join_order.json"), 'w') as f:
                json.dump(plans, f, indent=4)


def process_directory(subdir):
    meta_data_path = os.path.join(subdir, "meta_data.json")
    parameter_path = os.path.join(subdir, "parameter.json")
    plan_json_path = os.path.join(subdir, "plan.json")

    if os.path.isfile(meta_data_path) and os.path.isfile(parameter_path):
        print(f"Processing: {meta_data_path}")
        plans = generate_plans_for_query_by_distinct_plan(meta_data_path, parameter_path, plan_json_path)

        with open(os.path.join(subdir, "all_plans_by_join_order.json"), 'w') as f:
            json.dump(plans, f, indent=4)


def save_execution_plans_for_all_multiprocess(data_directory, num_processes=8):
    dirs = [x[0] for x in os.walk(data_directory) if 'a' in os.path.basename(x[0])]

    with Pool(num_processes) as pool:
        pool.map(process_directory, dirs)


if __name__ == "__main__":
    meta_data_path = '../training_data/JOB/'
    start_time = time.time()
    # meta_data_path = '../training_data/example_one/'
    save_execution_plans_for_all_multiprocess(meta_data_path)
    end_time = time.time()
    print(end_time - start_time)
