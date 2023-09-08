import numpy as np
import torch
import random


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

    # Greedy选择前k个plan保存
    selected_plans = []
    potential_plans = set(range(len(plans)))  # 直接使用所有的计划

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


def candidate_plan_selection_with_top_k(model, parameters, plans, k):
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


def candidate_plan_selection_with_random_sampling(model, parameters, plans, k, n):
    # 从所有的计划中随机采样n个
    sampled_plans = random.sample(plans, n)

    # 计算distance矩阵
    distances_matrix = []
    for param in parameters:
        param_embedding = model.parameter_net(param)
        distances = []
        for plan in sampled_plans:
            plan_embedding = model.plan_net.build_trees(plan)
            distance = torch.norm(plan_embedding - param_embedding).item()
            distances.append(distance)
        distances_matrix.append(distances)
    distances_matrix = np.array(distances_matrix)

    # Greedy选择前k个plan保存
    selected_plans = []
    potential_plans_indices = set(range(n))

    for step in range(k):
        min_total_distance = float('inf')
        next_selected_plan_index = None

        # 对于每个可能的计划，计算添加它后的总距离
        for plan_idx in potential_plans_indices:
            current_total_distance = 0
            for idx in range(len(parameters)):
                distances_to_selected = [distances_matrix[idx][i] for i in selected_plans]
                distances_to_selected.append(distances_matrix[idx][plan_idx])
                current_total_distance += min(distances_to_selected)

            if current_total_distance < min_total_distance:
                min_total_distance = current_total_distance
                next_selected_plan_index = plan_idx

        selected_plans.append(sampled_plans[next_selected_plan_index])
        potential_plans_indices.remove(next_selected_plan_index)

    return selected_plans


def candidate_plan_selection_random_sample_both(model, all_parameters, all_plans, k, m, n):

    sampled_parameters = random.sample(all_parameters, m)
    sampled_plans = random.sample(all_plans, n)

    # 计算distance矩阵
    distances_matrix = []
    for param in sampled_parameters:
        param_embedding = model.parameter_net(param)
        distances = []
        for plan in sampled_plans:
            plan_embedding = model.plan_net.build_trees(plan)
            distance = torch.norm(plan_embedding - param_embedding).item()
            distances.append(distance)
        distances_matrix.append(distances)
    distances_matrix = np.array(distances_matrix)

    # Greedy选择前k个plan保存
    selected_plans = []
    potential_plans_indices = set(range(n))

    for step in range(k):
        min_total_distance = float('inf')
        next_selected_plan_index = None

        # 对于每个可能的计划，计算添加它后的总距离
        for plan_idx in potential_plans_indices:
            current_total_distance = 0
            for idx in range(len(sampled_parameters)):
                distances_to_selected = [distances_matrix[idx][i] for i in selected_plans]
                distances_to_selected.append(distances_matrix[idx][plan_idx])
                current_total_distance += min(distances_to_selected)

            if current_total_distance < min_total_distance:
                min_total_distance = current_total_distance
                next_selected_plan_index = plan_idx

        selected_plans.append(sampled_plans[next_selected_plan_index])
        potential_plans_indices.remove(next_selected_plan_index)

    return selected_plans


