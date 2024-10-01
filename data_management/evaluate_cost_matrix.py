import os
import random
import time
import re

import psycopg2
from psycopg2 import errors
import json


from multiprocessing import Pool

try:
    import configure
except ImportError:
    from data_management import configure


time_regex = re.compile(r'Execution Time: ([\d\.]+) ms')


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
    cursor.execute(query_with_hint)
    plan = cursor.fetchone()
    cursor.close()
    cost = plan[0][0]['Plan']['Total Cost']
    return cost


def fetch_actual_latency(connection, query_with_hint, parameters):
    cursor = connection.cursor()
    cursor.execute("SET max_parallel_workers_per_gather TO 0;")

    query_with_hint = cursor.mogrify(query_with_hint, parameters).decode()

    start_time = time.perf_counter()
    cursor.execute(query_with_hint)
    end_time = time.perf_counter()

    cursor.close()
    latency = end_time - start_time
    return latency


def fetch_actual_latency_timeout(connection, query_with_hint, parameters, timeout):
    cursor = connection.cursor()
    # Set the statement timeout
    cursor.execute(f"SET statement_timeout TO {int(timeout * 1000)};")  # timeout in milliseconds
    cursor.execute("SET max_parallel_workers_per_gather TO 0;")

    query_with_hint = cursor.mogrify(query_with_hint, parameters).decode()

    start_time = time.perf_counter()
    try:
        cursor.execute(query_with_hint)
    except Exception as e:
        connection.rollback()
        return timeout * 10
        #print(f"Query terminated due to timeout: {e}")
    end_time = time.perf_counter()

    cursor.close()
    latency = end_time - start_time
    return latency

def fetch_actual_latency_bak(connection, query_with_hint, parameters):
    cursor = connection.cursor()
    cursor.execute("SET max_parallel_workers_per_gather TO 0;")

    query_with_hint = cursor.mogrify(query_with_hint, parameters).decode()

    #print(query_with_hint)

    try:
        # start_time = time.time()
        cursor.execute(query_with_hint)
        # end_time = time.time()
        explain_analyze_result = cursor.fetchall()

        latency = 50000.0
        for row in explain_analyze_result:
            # print(row[0])
            # print(type(row[0]))
            match = time_regex.search(row[0][0])
            if match:
                latency = float(match.group(1))
                break
    except errors.QueryCanceledError as e:
        connection.rollback()
        print("Query cancelled due to statement timeout.")
        return 50000.0

    cursor.close()
    # latency = end_time - start_time
    return latency


def get_counter_example(meta_data, plans, parameters_data):
    template = meta_data["template"]
    results = {}

    sampled_plan_keys = list(plans.keys())
    sampled_param_keys = list(parameters_data.keys())

    # print(f"{len(sampled_param_keys)} parameter vectors and {len(sampled_plan_keys)} plans")
    i = 0

    for param_key in sampled_param_keys:
        param_values = parameters_data[param_key]
        results[param_key] = {}

        for plan_key in sampled_plan_keys:
            connection = connect_to_pg()
            plan = plans[plan_key]
            plan_hint = generate_hint_from_plan(plan)
            query_with_hint = f"/*+ {plan_hint} */ EXPLAIN ANALYZE " + template
            # query_with_hint = f"/*+ {plan_hint} */ EXPLAIN (FORMAT JSON) " + template

            query_with_hint = query_with_hint.format(*param_values)

            # cost = fetch_plan_cost(connection, query_with_hint, param_values)
            st = time.time()
            cost = fetch_actual_latency(connection, query_with_hint, param_values)
            ed = time.time()
            results[param_key][plan_key] = ed - st
            connection.close()
            # print(f"count = {i}")
            i += 1

    return results


def evaluate_plans_for_parameters(connection, meta_data, plans, parameters_data):
    template = meta_data["template"]
    results = {}

    sampled_plan_keys = random.sample(list(plans.keys()), min(20, len(plans.keys())))
    sampled_param_keys = list(parameters_data.keys())[:4]

    # print(f"{len(sampled_param_keys)} parameter vectors and {len(sampled_plan_keys)} plans")
    i = 0

    for param_key in sampled_param_keys:
        param_values = parameters_data[param_key]
        results[param_key] = {}

        for plan_key in sampled_plan_keys:
            plan = plans[plan_key]
            plan_hint = generate_hint_from_plan(plan)
            query_with_hint = f"/*+ {plan_hint} */ EXPLAIN ANALYZE " + template
            # query_with_hint = f"/*+ {plan_hint} */ EXPLAIN (FORMAT JSON) " + template

            query_with_hint = query_with_hint.format(*param_values)

            # cost = fetch_plan_cost(connection, query_with_hint, param_values)
            cost = fetch_actual_latency(connection, query_with_hint, param_values)
            results[param_key][plan_key] = cost
            # print(f"count = {i}")
            i += 1

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

            with open(os.path.join(subdir, "cost_test.json"), 'w') as f_costs:
                json.dump(costs, f_costs, indent=4)

    connection.close()


def evaluate_directory(subdir):
    connection = connect_to_pg()

    with open(os.path.join(subdir, "meta_data.json"), 'r') as f_meta:
        meta_data = json.load(f_meta)

    with open(os.path.join(subdir, "hybrid_plans.json"), 'r') as f_plans:
        plans = json.load(f_plans)

    with open(os.path.join(subdir, "parameters.json"), 'r') as f_params:
        parameters = json.load(f_params)

    print(f"Processing {subdir}...")

    costs = evaluate_plans_for_parameters(connection, meta_data, plans, parameters)

    with open(os.path.join(subdir, "cost_matrix_2.json"), 'w') as f_costs:
        json.dump(costs, f_costs, indent=4)

    print(f"Finished {subdir}...")

    connection.close()


def evaluate_all_mutil_process(data_directory):
    directories_to_process = []

    # directories_to_process.append(data_directory + '14a')
    # # directories_to_process.append(data_directory + '18a')
    # directories_to_process.append(data_directory + '20a')
    # # directories_to_process.append(data_directory + '22a')

    for subdir, _, files in os.walk(data_directory):
        if "meta_data.json" in files:
            directories_to_process.append(subdir)

    with Pool(processes=12) as pool:
        pool.map(evaluate_directory, directories_to_process)


if __name__ == "__main__":
    meta_data_path = '../training_data/TPCDS/'
    start_time = time.time()
    evaluate_all_mutil_process(meta_data_path)
    end_time = time.time()
    print(end_time-start_time)
    # data_path = '../training_data/JOB/20a/'
    #
    # with open(os.path.join(data_path, "meta_data.json"), 'r') as f_meta:
    #     meta_data = json.load(f_meta)
    #
    # with open(os.path.join(data_path, "all_plans_by_hybrid_new.json"), 'r') as f_plans:
    #     plans = json.load(f_plans)
    #
    # parameters = {"parameter 1": [
    #     "cast",
    #     "complete",
    #     "Jenn's Grandmother",
    #     "drunk",
    #     "River Village Guard #1",
    #     "movie",
    #     "1885"
    # ],
    # "parameter 2": [
    #     "cast",
    #     "complete",
    #     "G\u00f6zde Barim",
    #     "Charles Griffey",
    #     "Church child",
    #     "movie",
    #     "2006"
    # ]}
    #
    #
    # results = get_counter_example(meta_data,plans,parameters)
    #
    # with open(os.path.join(data_path, "counter.json"), 'w') as f_costs:
    #     json.dump(results, f_costs, indent=4)


