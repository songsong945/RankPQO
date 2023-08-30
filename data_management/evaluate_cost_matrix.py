import os
import time

import psycopg2
import json

import configure


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

    def traverse_node(node, hints):
        node_type = node['Node Type']
        if 'Relation Name' in node:
            relation = node['Relation Name']
            hint = None
            if node_type == 'Seq Scan':
                hint = f"SeqScan({relation})"
            elif node_type == 'Index Scan':
                hint = f"IndexScan({relation})"
            if hint:
                hints.append(hint)

        if 'Plans' in node:
            for sub_node in node['Plans']:
                traverse_node(sub_node, hints)

    traverse_node(node, hints)
    return ', '.join(hints)


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
    cursor.fetchall()  # 确保读取所有结果，以便计算完整的查询执行时间
    end_time = time.time()

    cursor.close()
    latency = end_time - start_time
    return latency


def evaluate_plans_for_parameters(connection, meta_data, plans, parameters_data):
    template = meta_data["template"]
    results = {}

    for param_key, param_values in parameters_data.items():
        results[param_key] = {}
        for plan_key, plan in plans.items():
            plan_hint = generate_hint_from_plan(plan)
            query_with_hint = f"/*+ {plan_hint} */ " + template

            query_with_hint = query_with_hint.format(*param_values)

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

            with open(os.path.join(subdir, "cost_matrix.json"), 'w') as f_costs:
                json.dump(costs, f_costs, indent=4)

    connection.close()


if __name__ == "__main__":
    meta_data_path = '../training_data/JOB/'
    evaluate_all(meta_data_path)
