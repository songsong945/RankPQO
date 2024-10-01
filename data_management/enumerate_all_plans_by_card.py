import random
import time

import psycopg2
import json
import os
import configure

import numpy as np

from hybrid_plan_enumeration import get_structural_representation, compute_hash


def search_plan_for_alias(node, alias):
    if "Relation Name" in node and node["Alias"] == alias:
        return node["Plan Rows"]
    if "Plans" in node:
        for child in node["Plans"]:
            rows = search_plan_for_alias(child, alias)
            if rows:
                return rows
    return None


def get_row_count_candidates(row_count, exponent_base, exponent_range):
    min_exponent = -1 * min(np.log(row_count) // np.log(exponent_base), exponent_range)
    max_exponent = min_exponent + 2 * exponent_range
    candidates = row_count * np.power(float(exponent_base), np.arange(min_exponent, max_exponent + 1))
    candidates = np.array(list(map(lambda x: max(int(x), 1), candidates)))
    assert len(set(candidates)) == len(candidates)
    return candidates


def get_row_count_candidates_for_multiple(row_counts_dict, exponent_base=10, exponent_range=3):
    return {alias: get_row_count_candidates(row_count, exponent_base, exponent_range)
            for alias, row_count in row_counts_dict.items()}


def generate_hints_from_plan(plan, alias, exponent_base=10, exponent_range=2):
    rows = search_plan_for_alias(plan["Plan"], alias)
    if rows:
        candidates = get_row_count_candidates(rows, exponent_base, exponent_range)
        hints = [f"/*+ Rows({alias}, {candidate}) */" for candidate in candidates]
        return hints
    else:
        return []


def search_plan_for_aliases(node, aliases):
    result = {alias: None for alias in aliases}

    if "Relation Name" in node and node["Alias"] in aliases:
        result[node["Alias"]] = node["Plan Rows"]

    if "Plans" in node:
        for child in node["Plans"]:
            child_result = search_plan_for_aliases(child, aliases)
            for alias, rows in child_result.items():
                if rows is not None:
                    result[alias] = rows
                else:
                    result[alias] = 1

    return result


def generate_hints_from_plan_with_sampling(plan, aliases, k=20):
    rows_dict = search_plan_for_aliases(plan["Plan"], aliases)
    candidates_dict = get_row_count_candidates_for_multiple(rows_dict)

    hints_list = []
    seen_combinations = set()

    for _ in range(k):
        while True:
            sampled_counts = {alias: np.random.choice(candidates) for alias, candidates in candidates_dict.items()}
            hashable_combination = tuple((k, v) for k, v in sorted(sampled_counts.items()))
            if hashable_combination not in seen_combinations:
                seen_combinations.add(hashable_combination)
                break

        hint_content = '\n'.join([f"Rows({alias}, {row_count})" for alias, row_count in sampled_counts.items()])
        hint = f"/*+ {hint_content} */"
        hints_list.append(hint)

    return hints_list


def connect_to_pg():
    connection = psycopg2.connect(
        dbname=configure.dbname3,
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
    seen_hashes = set()

    table_aliases = [predicate["alias"] for predicate in meta_data["predicates"]]

    idx = 0
    i = 0
    for params in enumerate(parameters_list.values()):

        i += 1

        if i > 10:
            break

        plan = fetch_execution_plan(connection, meta_data['template'], params[1])
        representation = get_structural_representation(plan['Plan'])
        hash_val = compute_hash(representation)
        if hash_val in seen_hashes:
            continue
        seen_hashes.add(hash_val)
        idx += 1
        plans[f"plan {idx}"] = plan

        # for alias in table_aliases:
        #     hints = generate_hints_from_plan(plan, alias)
        for _ in range(3):
            plan = random.choice(list(plans.values()))
            hints = generate_hints_from_plan_with_sampling(plan, table_aliases)
            #print(len(hints))
            print(hints[0])
            for hint in hints:

                modified_plan_with_hint = fetch_execution_plan(connection, hint + " " + meta_data['template'],
                                                               params[1])
                representation = get_structural_representation(modified_plan_with_hint['Plan'])
                hash_val = compute_hash(representation)
                if hash_val in seen_hashes:
                    continue
                seen_hashes.add(hash_val)
                idx += 1
                plans[f"modified {idx}"] = modified_plan_with_hint

    connection.close()
    # print(len(plans))
    return plans


# 4. Save execution plans as JSON
def save_execution_plans_for_all(data_directory):
    total_time = 0
    num = 0
    for subdir, _, _ in os.walk(data_directory):
        meta_data_path = os.path.join(subdir, "meta_data.json")
        parameter_path = os.path.join(subdir, "parameters.json")

        print(subdir)

        if os.path.isfile(meta_data_path) and os.path.isfile(parameter_path):
            # print(f"Processing: {meta_data_path}")
            start_time = time.time()
            plans = generate_plans_for_query(meta_data_path, parameter_path)
            end_time = time.time()

            num += len(plans)

            total_time += (end_time - start_time)

            with open(os.path.join(subdir, f"kepler_plans.json"), 'w') as f:
                json.dump(plans, f, indent=4)

    print(f'plan number: {num}')
    print(f'time: {total_time}')


if __name__ == "__main__":
    meta_data_path = '../training_data/TPCDS/'
    # meta_data_path = '../training_data/example_one/'
    save_execution_plans_for_all(meta_data_path)
