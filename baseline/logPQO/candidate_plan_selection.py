import json
import os
import time

import numpy as np

def candidate_plan_selection(parameters, plans, k, costs):
    # 计算distance矩阵
    distances_matrix = []

    for parameter in parameters.keys():
        distances = []
        for plan in plans.keys():
            distances.append(costs[parameter][plan])
        distances_matrix.append(distances)
    distances_matrix = np.array(distances_matrix)

    # Greedy选择前k个plan保存
    selected_plans = []
    potential_plans = list(range(len(plans)))

    for step in range(k):
        min_total_distance = float('inf')
        next_selected_plan = None

        for plan_idx in potential_plans:
            current_total_distance = 0
            for idx in range(len(parameters)):
                distances_to_selected = [distances_matrix[idx][i] for i in selected_plans]
                distances_to_selected.append(distances_matrix[idx][plan_idx])
                current_total_distance += min(distances_to_selected)

            if current_total_distance < min_total_distance:
                min_total_distance = current_total_distance
                next_selected_plan = plan_idx

        if next_selected_plan is None:
            break

        selected_plans.append(next_selected_plan)
        potential_plans.remove(next_selected_plan)

    return selected_plans

if __name__  == "__main__":
    training_data = '../../training_data/JOB/'
    all_folders = []
    for subdir, _, files in os.walk(training_data):
        if 'a' in os.path.basename(subdir):
            all_folders.append(os.path.basename(subdir))

    for k in [10, 20, 30, 40, 50]:

        total_time = 0

        for template_id in all_folders:
            path = os.path.join(training_data, template_id)

            with open(os.path.join(path, "cost_matrix_pg.json"), 'r') as f_cost:
                costs = json.load(f_cost)

            with open(os.path.join(path, "plan_pg.json"), 'r') as f_plans:
                plans = json.load(f_plans)

            with open(os.path.join(path, "parameter_new.json"), 'r') as f_params:
                parameters = json.load(f_params)

            #print(f"Processing {path}...")

            start_time = time.time()
            candidate_plans = candidate_plan_selection(parameters, plans, k, costs)
            end_time = time.time()

            total_time += (end_time - start_time)

            selected_plan_dict = {}
            all_plan_keys = list(plans.keys())
            for idx in candidate_plans:
                key = all_plan_keys[idx]
                selected_plan_dict[key] = plans[key]

            with open(os.path.join(path, f"selected_plan_baseline1_{k}.json"), 'w') as f:
                json.dump(selected_plan_dict, f, indent=4)

        print(f"select {k} plans with {total_time}s")
