import json
import os

from evaluate_cost_matrix import generate_hint_from_plan, fetch_plan_cost, connect_to_pg
from logPQO.train import _param_path, _cost_path, _plan_path

def get_what_if_cost(connection, meta_data, plans, parameters_data):
    template = meta_data["template"]
    results = {}

    sampled_plan_keys = plans.keys()
    sampled_param_keys = parameters_data.keys()

    for param_key in sampled_param_keys:
        param_values = parameters_data[param_key]
        results[param_key] = {}

        for plan_key in sampled_plan_keys:
            plan = plans[plan_key]
            plan_hint = generate_hint_from_plan(plan)
            query_with_hint = f"/*+ {plan_hint} */ EXPLAIN ANALYZE " + template

            query_with_hint = query_with_hint.format(*param_values)

            cost = fetch_plan_cost(connection, query_with_hint, param_values)
            results[param_key][plan_key] = cost

    return results


if __name__  == "__main__":
    training_data = '../../training_data/JOB/'
    all_folders = []
    for subdir, _, files in os.walk(training_data):
        if ('a' in os.path.basename(subdir) and "meta_data.json" in files and "all_plans_by_hybrid_new.json" in files
                and "parameter_new.json" in files):
            all_folders.append(os.path.basename(subdir))

    for template_id in all_folders:

        path = os.path.join(training_data, template_id)

        connection = connect_to_pg()

        with open(os.path.join(path, "meta_data.json"), 'r') as f_meta:
            meta_data = json.load(f_meta)

        with open(os.path.join(path, "all_plans_by_hybrid_new.json"), 'r') as f_plans:
            plans = json.load(f_plans)

        with open(os.path.join(path, "parameter_new.json"), 'r') as f_params:
            parameters = json.load(f_params)

        print(f"Processing {path}...")

        costs = get_what_if_cost(connection, meta_data, plans, parameters)

        with open(os.path.join(path, "cost_matrix_baseline1.json"), 'w') as f_costs:
            json.dump(costs, f_costs, indent=4)
