import numpy as np
import torch


def best_plan_selection(model, parameter, plans):
    model.parameter_net.eval()
    model.plan_net.eval()

    param_embedding = model.parameter_net(parameter)

    distances = []
    for plan in plans:
        tree_plan = model.plan_net.build_trees(plan)
        plan_embedding = model.plan_net(tree_plan)
        distance = torch.norm(plan_embedding - param_embedding).item()
        distances.append(distance)

    sorted_plan_indices = np.argsort(distances)

    optimal_plan = plans[sorted_plan_indices[0]]

    return optimal_plan


def candidate_plan_selection(model, parameters, plans, k):
    # 计算distance矩阵
    distances_matrix = []
    for param in parameters:
        param_embedding = model.parameter_net(param)
        distances = []
        for plan in plans:
            plan_embedding = model.plan_net.build_trees(plan)
            distance = torch.norm(plan_embedding - param_embedding).item()
            distances.append(distance)
        distances_matrix.append(distances)
    distances_matrix = np.array(distances_matrix)

    # 每个参数选择前k个plan
    all_top_k_indices_for_params = [list(np.argsort(dist)[:k]) for dist in distances_matrix]

    # Greedy选择前k个plan保存
    selected_plans = []
    potential_plans = set([plan for sublist in all_top_k_indices_for_params for plan in sublist])

    for step in range(k):
        min_total_distance = float('inf')
        next_selected_plan = None

        # 对于每个可能的计划，计算添加它后的总距离
        for plan_idx in potential_plans:
            current_total_distance = 0
            for idx in range(len(parameters)):
                distances_to_selected = [distances_matrix[idx][i] for i in selected_plans]
                distances_to_selected.append(distances_matrix[idx][plan_idx])
                current_total_distance += min(distances_to_selected)

            if current_total_distance < min_total_distance:
                min_total_distance = current_total_distance
                next_selected_plan = plan_idx

        selected_plans.append(next_selected_plan)
        potential_plans.remove(next_selected_plan)

    return selected_plans


