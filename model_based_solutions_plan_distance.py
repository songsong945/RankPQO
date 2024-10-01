import numpy as np
import torch
import random

from model_plan_distance import concatenate_z_to_nodes


def best_plan_selection(model, parameter, plans, device):

    parameter_list = np.expand_dims(parameter, 0)
    parameter_x = torch.tensor(parameter_list).to(device)
    param_embedding = model.parameter_net(parameter_x)

    plan_x = model.plan_net.build_trees(plans)
    plan_x_z = concatenate_z_to_nodes(plan_x, param_embedding)
    distances = model.plan_net(plan_x_z)


    distances = distances.detach().cpu().numpy()
    distances = distances.flatten()
    sorted_plan_indices = np.argsort(distances)

    optimal_plan = sorted_plan_indices[-1]

    return optimal_plan


def candidate_plan_selection(model, parameters, plans, k, device):

    distances_matrix = []

    parameter_x = torch.tensor(parameters).to(device)
    param_embeddings = model.parameter_net(parameter_x)
    plans = model.plan_net.build_trees(plans)

    for param_embedding in param_embeddings:
        if param_embedding.dim() == 1:
            param_embedding = param_embedding.unsqueeze(0)
        elif param_embedding.dim() == 3 and param_embedding.shape[2] == 1:
            param_embedding = param_embedding.squeeze(2)

        plan_x_z = concatenate_z_to_nodes(plans, param_embedding)
        distances = model.plan_net(plan_x_z)

        distances = distances.detach().cpu().numpy()
        distances_matrix.append(distances)

        # 显式删除变量，释放显存
        del plan_x_z, distances
        torch.cuda.empty_cache()

    distances_matrix = np.array(distances_matrix)

    selected_plans = []
    potential_plans = set(range(len(plans)))

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
