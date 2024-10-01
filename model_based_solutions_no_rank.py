import numpy as np
import torch
import random


def best_plan_selection(model, parameter, plans, device):
    # model.parameter_net.eval()
    # model.plan_net.eval()
    parameter_list = np.expand_dims(parameter, 0)
    parameter_x = torch.tensor(parameter_list).to(device)
    param_embedding = model.parameter_net(parameter_x)

    plan_x = model.plan_net.build_trees(plans)
    plan_embeddings = model.plan_net(plan_x)

    distances = []
    for plan_embedding in plan_embeddings:
        if plan_embedding.dim() == 1:
            plan_embedding = plan_embedding.unsqueeze(0)
        distance = model.cost_est_net(torch.cat((plan_embedding, param_embedding), dim=-1)).float()
        distance = distance.detach().cpu().numpy()
        distances.append(distance)

    distances = np.array(distances)
    distances = distances.flatten()

    sorted_plan_indices = np.argsort(distances)

    optimal_plan = sorted_plan_indices[0]

    #print(optimal_plan)

    return optimal_plan


def candidate_plan_selection(model, parameters, plans, k, device):
    # 计算distance矩阵
    distances_matrix = []

    parameter_x = torch.tensor(parameters).to(device)
    param_embeddings = model.parameter_net(parameter_x)

    plan_x = model.plan_net.build_trees(plans)
    plan_embeddings = model.plan_net(plan_x)

    for param_embedding in param_embeddings:
        distances = []
        for plan_embedding in plan_embeddings:
            distance = model.cost_est_net(torch.cat((plan_embedding, param_embedding), dim=-1)).float()
            distance = distance.detach().cpu()
            distances.append(distance)
        distances_matrix.append(torch.stack(distances).numpy())
    distances_matrix = np.array(distances_matrix)

    # Greedy选择前k个plan保存
    selected_plans = []
    potential_plans = set(range(len(plans)))  # 直接使用所有的计划

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

