import json
import logging
import os
import random
import time
import psycopg2
import re
from psycopg2 import errors

from data_management import configure
from data_management.evaluate_cost_matrix import generate_hint_from_plan, fetch_actual_latency
from model.model_no_rank import RankPQOModel
from model_based_solutions_no_rank import candidate_plan_selection, best_plan_selection
from train.train_no_rank import get_param_info, _meta_path, _plan_path, _param_path

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s')

time_regex = re.compile(r'Execution Time: ([\d\.]+) ms')


def _selected_plan_path(input, base):
    return os.path.join(base, input)


def _cost_path(base):
    return os.path.join(base, "latency_matrix_new.json")


def connect_to_pg_tpcds():
    connection = psycopg2.connect(
        dbname=configure.dbname3,
        user=configure.user,
        password=configure.password,
        host=configure.host,
        port=configure.port
    )
    return connection

def connect_to_pg_job():
    connection = psycopg2.connect(
        dbname=configure.dbname,
        user=configure.user,
        password=configure.password,
        host=configure.host,
        port=configure.port
    )
    return connection


def connect_to_pg():
    connection1 = psycopg2.connect(
        dbname=configure.dbname,
        user=configure.user,
        password=configure.password,
        host=configure.host,
        port=configure.port,
        options='-c statement_timeout=5000'
    )
    connection2 = psycopg2.connect(
        dbname=configure.dbname,
        user=configure.user,
        password=configure.password,
        host=configure.host,
        port=configure.port
    )
    return connection1, connection2


def fetch_actual_latency2(connection1, connection2, query_with_hint, parameters, query):
    cursor = connection1.cursor()
    cursor.execute("SET max_parallel_workers_per_gather TO 0;")

    query_with_hint = cursor.mogrify(query_with_hint, parameters).decode()
    # print(query_with_hint)

    try:
        # start_time = time.time()
        cursor.execute(query_with_hint)
        # end_time = time.time()
        explain_analyze_result = cursor.fetchall()

        latency = 5000.0
        for row in explain_analyze_result:
            match = time_regex.search(row[0])
            if match:
                latency = float(match.group(1))
                break
    except errors.QueryCanceledError as e:
        connection1.rollback()
        cursor.close()

        cursor = connection2.cursor()
        cursor.execute("SET max_parallel_workers_per_gather TO 0;")

        query = cursor.mogrify(query, parameters).decode()
        cursor.execute(query)
        explain_analyze_result = cursor.fetchall()

        latency = 5000.0
        for row in explain_analyze_result:
            match = time_regex.search(row[0])
            if match:
                latency += float(match.group(1))
                break
    cursor.close()
    return latency


def candidate_selection(data, is_share, model_path, device, k, output):
    # all_folders = ['query072', 'query025', 'query100']
    all_folders = []
    for subdir, _, files in os.walk(data):
        if "meta_data.json" in files:
            all_folders.append(os.path.basename(subdir))

    print(all_folders)
    total_time = 0

    for folder in all_folders:
        template_id = folder
        if is_share:
            model_path_file = model_path
        else:
            model_path_file = os.path.join(model_path, folder)

        path = os.path.join(data, template_id)

        with open(_meta_path(path), 'r') as f:
            meta = json.load(f)

        with open(_plan_path(path), 'r') as f:
            plan = json.load(f)

        with open(_param_path(path), 'r') as f:
            param = json.load(f)

        # print(f"Process {folder}")

        params, preprocess_info = get_param_info(meta)
        plans = list(plan.values())
        parameters = list(param.values())
        print(f"plan number: {len(plans)}")
        rank_PQO_model = RankPQOModel(None, template_id, preprocess_info, device=device)
        rank_PQO_model.load(model_path_file, fist_layer=1)
        feature_generator = rank_PQO_model._feature_generator
        plans = feature_generator.transform(plans)
        parameters = feature_generator.transform_z(parameters, params, preprocess_info)
        start_time = time.time()
        selected_plans = candidate_plan_selection(rank_PQO_model, parameters, plans, k, device)
        end_time = time.time()

        selected_plan_dict = {}
        all_plan_keys = list(plan.keys())
        for idx in selected_plans:
            key = all_plan_keys[idx]
            selected_plan_dict[key] = plan[key]

        selected_plans_path = os.path.join(path, output)
        with open(selected_plans_path, 'w') as f:
            json.dump(selected_plan_dict, f, indent=4)

        # print(f"Selected plans saved to {selected_plans_path}")
        total_time += (end_time - start_time)
    print(f"Selected plans time {total_time}s")


def best_plan_prediction2(data, is_share, model_path, device, input, output):
    train_predict_time_total = 0.0
    train_execution_time_total = 0.0
    test_predict_time_total = 0.0
    test_execution_time_total = 0.0
    results = {}

    # all_folders = ['query099', 'query040', 'query027', 'query084', 'query085', 'query025', 'query013', 'query100',
    #                'query101', 'query018', 'query091', 'query019', 'query050']
    #
    # all_folders = ['query091', 'query019', 'query050']
    # all_folders = ['query025']
    all_folders = []

    for subdir, _, files in os.walk(data):
        if "meta_data.json" in files:
            all_folders.append(os.path.basename(subdir))

    # connection1, connection2 = connect_to_pg()
    # connection = connect_to_pg_tpcds()
    connection = connect_to_pg_job()

    for folder in all_folders:
        print(f"Processing {folder}")
        execution_time_template = 0.0
        template_id = folder
        if is_share:
            model_path_file = model_path
        else:
            model_path_file = os.path.join(model_path, folder)

        path = os.path.join(data, template_id)

        with open(_meta_path(path), 'r') as f:
            meta = json.load(f)

        with open(_selected_plan_path(input, path), 'r') as f:
            plan = json.load(f)

        with open(_param_path(path), 'r') as f:
            param = json.load(f)

        with open(_cost_path(path), 'r') as f:
            cost = json.load(f)


        train_keys_list = list(cost.keys())[:160]
        # test_keys_list = list(cost.keys())[160:200]
        # train_keys_list = list(param.keys())[:4]
        # test_keys_list = list(param.keys())[4:5]

        template = meta["template"]

        params, preprocess_info = get_param_info(meta)
        plans = list(plan.values())
        parameters_train_data = [param[key] for key in train_keys_list if key in param]
        # parameters_test_data = [param[key] for key in test_keys_list if key in param]

        rank_PQO_model = RankPQOModel(None, template_id, preprocess_info, device=device)
        rank_PQO_model.load(model_path_file, fist_layer=1)
        feature_generator = rank_PQO_model._feature_generator
        plans_t = feature_generator.transform(plans)
        train_parameters = feature_generator.transform_z(parameters_train_data, params, preprocess_info)
        # test_parameters = feature_generator.transform_z(parameters_test_data, params, preprocess_info)

        print(f"Processing training queries")

        for i, parameter in enumerate(train_parameters):
            start_time = time.time()
            best_plan_id = best_plan_selection(rank_PQO_model, parameter, plans_t, device)
            end_time = time.time()
            predict_time = end_time - start_time

            train_predict_time_total += predict_time

            best_plan = plans[best_plan_id]
            param_values = parameters_train_data[i]
            plan_hint = generate_hint_from_plan(best_plan)
            # query_with_hint = f"/*+ {plan_hint} */ EXPLAIN " + template
            query_with_hint = f"/*+ {plan_hint} */ EXPLAIN ANALYZE " + template

            query_with_hint = query_with_hint.format(*param_values)

            # query = f"/*+ */ EXPLAIN ANALYZE " + template
            #
            # query = query.format(*parameter)

            # latency = fetch_actual_latency2(connection1, connection2, query_with_hint, param_values, query)
            latency = fetch_actual_latency(connection, query_with_hint, param_values)

            train_execution_time_total += (latency)
            execution_time_template += (latency + predict_time)

            # if latency / 1000 > 5:
            logging.info(f"Query {folder}-{i}: plan select {predict_time}s and execute {latency} s")

        print(f"Processing testing queries")

        # for i, parameter in enumerate(test_parameters):
        #     start_time = time.time()
        #     best_plan_id = best_plan_selection(rank_PQO_model, parameter, plans_t, device)
        #     end_time = time.time()
        #     predict_time = end_time - start_time
        #
        #     test_predict_time_total += predict_time
        #
        #     best_plan = plans[best_plan_id]
        #     param_values = parameters_test_data[i]
        #     plan_hint = generate_hint_from_plan(best_plan)
        #     query_with_hint = f"/*+ {plan_hint} */ EXPLAIN ANALYZE " + template
        #
        #     query_with_hint = query_with_hint.format(*param_values)
        #
        #     # query = f"/*+ */ EXPLAIN ANALYZE " + template
        #     #
        #     # query = query.format(*parameter)
        #
        #     # latency = fetch_actual_latency2(connection1, connection2, query_with_hint, param_values, query)
        #     latency = fetch_actual_latency(connection, query_with_hint, param_values)
        #
        #     test_execution_time_total += (latency)
        #     execution_time_template += (latency + predict_time)
        #
        #     # if latency / 1000 > 5:
        #     logging.info(f"Query {folder}-{i}: plan select {predict_time}s and execute {latency} s")

        results[folder] = execution_time_template
        logging.info(f"Query {folder} time: {execution_time_template}s")

    # logging.info(f"total train predict time: {train_predict_time_total}s")
    # logging.info(f"total train execution time: {train_execution_time_total}s")
    # logging.info(f"total test predict time: {test_predict_time_total}s")
    # logging.info(f"total test execution time: {test_execution_time_total}s")
    logging.info("result time")
    print(train_predict_time_total)
    print(train_execution_time_total)
    print(test_predict_time_total)
    print(test_execution_time_total)

    results['total execution time'] = (train_predict_time_total + train_execution_time_total
                                       + test_predict_time_total + test_execution_time_total)
    print(results['total execution time'])

    with open(output, "w") as file:
        for key, values in results.items():
            file.write(f"{key}: {str(values)}\n")


def best_plan_prediction(data, is_share, model_path, device, input, output):
    predict_time_total = 0.0
    execution_time_total = 0.0
    results = {}

    all_folders = []
    for subdir, _, files in os.walk(data):
        folder_name = os.path.basename(subdir)
        if folder_name:
            all_folders.append(folder_name)

    # all_folders = ["18a"]

    connection1, connection2 = connect_to_pg()

    for folder in all_folders:
        execution_time_template = 0.0
        template_id = folder
        if is_share:
            model_path_file = model_path
        else:
            model_path_file = os.path.join(model_path, folder)

        path = os.path.join(data, template_id)

        with open(_meta_path(path), 'r') as f:
            meta = json.load(f)

        with open(_selected_plan_path(input, path), 'r') as f:
            plan = json.load(f)

        with open(_param_path(path), 'r') as f:
            param = json.load(f)

        template = meta["template"]

        params, preprocess_info = get_param_info(meta)
        plans = list(plan.values())
        parameters_data = list(param.values())
        rank_PQO_model = RankPQOModel(None, template_id, preprocess_info, device=device)
        rank_PQO_model.load(model_path_file, fist_layer=1)
        feature_generator = rank_PQO_model._feature_generator
        plans_t = feature_generator.transform(plans)
        parameters = feature_generator.transform_z(parameters_data, params, preprocess_info)
        # print(f"Processing {folder}")
        for i, parameter in enumerate(parameters):

            # if i >= 50:
            #     break

            # print(parameters_data[i])

            start_time = time.time()
            best_plan_id = best_plan_selection(rank_PQO_model, parameter, plans_t, device)
            end_time = time.time()
            predict_time = end_time - start_time

            # logging.info(f"best plan: {best_plan_id}")

            predict_time_total += predict_time

            # logging.info(f"Query {folder}-{i}: plan select {predict_time}s")

            best_plan = plans[best_plan_id]
            # print(best_plan)
            param_values = parameters_data[i]
            plan_hint = generate_hint_from_plan(best_plan)
            # plan_hint = ""
            query_with_hint = f"/*+ {plan_hint} */ EXPLAIN ANALYZE " + template

            query_with_hint = query_with_hint.format(*param_values)

            query = f"/*+ */ EXPLAIN ANALYZE " + template

            query = query.format(*parameter)

            latency = fetch_actual_latency2(connection1, connection2, query_with_hint, param_values, query)

            execution_time_total += (latency / 1000)
            execution_time_template += (latency / 1000 + predict_time)

            if latency / 1000 > 5:
                logging.info(f"Query {folder}-{i}: plan select {predict_time}s and execute {latency / 1000} s")

        results[folder] = execution_time_template / 1000

    logging.info(f"total predict time: {predict_time_total}s")
    logging.info(f"total execution time: {execution_time_total}s")

    results['total execution time'] = execution_time_total

    with open(output, "w") as file:
        for key, values in results.items():
            file.write(f"{key}: {str(values)}\n")


def PG_original2(data, output):
    train_execution_time_total = 0.0
    test_execution_time_total = 0.0
    results = {}

    all_folders = []
    for subdir, _, files in os.walk(data):
        if 'a' in os.path.basename(subdir) and "meta_data.json" in files:
            all_folders.append(os.path.basename(subdir))

    _, connection = connect_to_pg()

    for folder in all_folders:
        template_id = folder
        execution_time_template = 0.0

        path = os.path.join(data, template_id)

        with open(_meta_path(path), 'r') as f:
            meta = json.load(f)

        with open(_param_path(path), 'r') as f:
            param = json.load(f)

        with open(_cost_path(path), 'r') as f:
            cost = json.load(f)

        train_keys_list = list(cost.keys())[:160]
        test_keys_list = list(cost.keys())[160:200]

        template = meta["template"]
        print(f"Processing {folder}")

        for param_key in train_keys_list:
            parameter = param[param_key]

            query = f"/*+ */ EXPLAIN ANALYZE " + template

            query = query.format(*parameter)

            latency = fetch_actual_latency(connection, query, parameter)

            logging.info(f"Query {folder}-{param_key}: execute {latency}s")

            train_execution_time_total += latency / 1000

            execution_time_template += latency / 1000

        for param_key in test_keys_list:
            parameter = param[param_key]

            query = f"/*+ */ EXPLAIN ANALYZE " + template

            query = query.format(*parameter)

            latency = fetch_actual_latency(connection, query, parameter)

            logging.info(f"Query {folder}-{param_key}: execute {latency}s")

            test_execution_time_total += latency / 1000

            execution_time_template += latency / 1000

        #     if latency / 1000 > 2:
        #         logging.info(f"Query {folder}-{i}: execute {latency / 1000}s")
        #
        # logging.info(f"Query {folder} avg execution time: {execution_time_template / 1000}s")

        results[folder] = execution_time_template / 1000

    logging.info(f"total train execution time: {train_execution_time_total * 1000}s")
    logging.info(f"total test execution time: {test_execution_time_total * 1000}s")

    results['total train execution time'] = train_execution_time_total * 1000
    results['total test execution time'] = test_execution_time_total * 1000

    with open(output, "w") as file:
        for key, values in results.items():
            file.write(f"{key}: {str(values)}\n")


def PG_original(data, output):
    execution_time_total = 0.0
    results = {}

    # all_folders = ['query025']
    all_folders = []
    for subdir, _, files in os.walk(data):
        if "meta_data.json" in files:
            all_folders.append(os.path.basename(subdir))

    connection = connect_to_pg_job()
    # connection = connect_to_pg_tpcds()

    for folder in all_folders:
        template_id = folder
        execution_time_template = 0.0

        path = os.path.join(data, template_id)

        with open(_meta_path(path), 'r') as f:
            meta = json.load(f)

        with open(_param_path(path), 'r') as f:
            param = json.load(f)

        template = meta["template"]

        parameters_data = list(param.values())
        print(f"Processing {folder}")
        for i, parameter in enumerate(parameters_data):
            if i >= 200:
                break
            query = f"/*+ */ EXPLAIN " + template
            # query = f"/*+ */ EXPLAIN ANALYZE " + template
            # print(template)
            # print(parameter)
            # query = f"/*+ */ EXPLAIN " + template

            query = query.format(*parameter)

            latency = fetch_actual_latency(connection, query, parameter)

            execution_time_total += latency

            execution_time_template += latency

            #     if latency / 1000 > 2:
            # logging.info(f"Query {folder}-{i}: execute {latency}s")
        #
        logging.info(f"Query {folder} avg execution time: {execution_time_template/200 }s")

        results[folder] = execution_time_template / 200

    logging.info(f"total execution time: {execution_time_total}s")

    results['total execution time'] = execution_time_total

    with open(output, "w") as file:
        for key, values in results.items():
            file.write(f"{key}: {str(values)}\n")


if __name__ == '__main__':
    candidate_selection("./training_data/JOB/", True,
                        "./checkpoints/A_N_JOB_alternating_3k_1/", "cuda:2", 30,
                        f"A_N_JOB_selected_plans_3k_1_30.json")

    best_plan_prediction2("./training_data/JOB/", True,
                          "./checkpoints/A_N_JOB_alternating_3k_1/", "cuda:2",
                          f"A_N_JOB_selected_plans_3k_1_30.json", f"./results/JOB/A_N_JOB_30.txt")

    # print('random on graph')
    # PG_original2("./training_data/JOB/", "./results/JOB/PG.txt")
    # candidate_selection("./training_data/JOB/", True,
    #                     "./checkpoints/A_D_JOB_alternating_3k_1/", "cuda:2", 30,
    #                     f"A_D_JOB_selected_plans_3k_1_30.json")
    # candidate_selection("./training_data/TPCDS/", True,
    #                     "./checkpoints/TPCDS_alternating_3k_1/", "cuda:2", 30,
    #                     f"selected_plans_30.json")
    # best_plan_prediction2("./training_data/TPCDS/", True,
    #                       "./checkpoints/TPCDS_alternating_3k_1/", "cuda:2",
    #                       f"selected_plans_30.json", f"./results/TPCDS/RankPQO_hybrid_30.txt")
    # PG_original("./training_data/JOB/", "./results/JOB/PG.txt")
    # best_plan_prediction2("./training_data/JOB/", True,
    #                       "./checkpoints/A_D_JOB_alternating_3k_1/", "cuda:2",
    #                       f"A_D_JOB_selected_plans_3k_1_30.json", f"./results/JOB/A_D_JOB_30.txt")
    # for k in [40, 50]:
    # for k in [10, 20, 30, 40, 50]:
    #     print(k)
    #     candidate_selection("./training_data/JOB/", False,
    #                         "./checkpoints/JOB_no_share_new_3k_10ep/", "cuda:2", k, f"selected_plans_3k_no_share_{k}.json")
    #     best_plan_prediction2("./training_data/JOB/", False,
    #                          "./checkpoints/JOB_no_share_new_3k_10ep/", "cuda:2",
    #                          f"selected_plans_3k_no_share_{k}.json", f"./results/JOB/RankPQO_3k_no_share_{k}.txt")
    #
    # for k in [10, 20, 30, 40, 50]:
    #     print(k)
    #     candidate_selection("./training_data/JOB/", True,
    #                         "./checkpoints/JOB_alternating_new_3k_10ep_1step/", "cuda:2", k, f"selected_plans_3k_1_{k}.json")
    #     best_plan_prediction2("./training_data/JOB/", True,
    #                          "./checkpoints/JOB_alternating_new_3k_10ep_1step/", "cuda:2",
    #                          f"selected_plans_3k_1_{k}.json", f"./results/JOB/RankPQO_3k_1_{k}.txt")

    # vary epochs
    # for k in [1, 5, 15, 20]:
    #     print(k)
    #     candidate_selection("./training_data/JOB/", False,
    #                         f"./checkpoints/JOB_no_share_new_3k_{k}ep/", "cuda:2", 30,
    #                         f"selected_plans_3k_no_share_30_{k}ep.json")
    #     best_plan_prediction2("./training_data/JOB/", False,
    #                           f"./checkpoints/JOB_no_share_new_3k_{k}ep/", "cuda:2",
    #                           f"selected_plans_3k_no_share_30_{k}ep.json", f"./results/JOB/RankPQO_3k_no_share_30_{k}ep.txt")

    # for k in [1, 5, 15, 20]:
    #     print(k)
    #     candidate_selection("./training_data/JOB/", True,
    #                         f"./checkpoints/JOB_alternating_new_3k_{k}ep_1step/", "cuda:2", 30,
    #                         f"selected_plans_3k_1_30_{k}ep.json")
    #     best_plan_prediction2("./training_data/JOB/", True,
    #                           f"./checkpoints/JOB_alternating_new_3k_{k}ep_1step/", "cuda:2",
    #                           f"selected_plans_3k_1_30_{k}ep.json", f"./results/JOB/RankPQO_3k_1_30_{k}ep.txt")

    # all_plans_by_card_{k}.json

    # PG_original2("./training_data/JOB/", "./results/JOB/PG_train_test.txt")
    # PG_original("./training_data/TPCDS/", "./results/TPCDS/PG.txt")
    # PG_original("./training_data/job_pqo_resampled/", "./results/job_pqo_resampled/PG.txt")

    # vary data size
    # for s in ['1k', '2k', '4k', '5k']:
    # # for s in ['1k', '2k', '20k', '25k']:
    #     print(s)
    #     candidate_selection("./training_data/JOB/", False,
    #                         f"./checkpoints/JOB_no_share_new_{s}_10ep/", "cuda:1", 30, f"selected_plans_{s}_no_share_30.json")
    #     best_plan_prediction2("./training_data/JOB/", False,
    #                          f"./checkpoints/JOB_no_share_new_{s}_10ep/", "cuda:1",
    #                          f"selected_plans_{s}_no_share_30.json", f"./results/JOB/RankPQO_{s}_no_share_30.txt")

    # for s in ['1k', '2k', '4k', '5k']:
    # # for s in ['1k', '2k', '20k', '25k']:
    #     print(s)
    #     candidate_selection("./training_data/JOB/", True,
    #                         f"./checkpoints/JOB_alternating_new_{s}_10ep_1step/", "cuda:1", 30, f"selected_plans_{s}_1_30.json")
    #     best_plan_prediction2("./training_data/JOB/", True,
    #                          f"./checkpoints/JOB_alternating_new_{s}_10ep_1step/", "cuda:1",
    #                          f"selected_plans_{s}_1_30.json", f"./results/JOB/RankPQO_{s}_1_30.txt")

    # vary step
    # for step in [10, 5, 2]:
    # #for step in [50,25]:
    #     print(step)
    #     candidate_selection("./training_data/JOB/", True,
    #                         f"./checkpoints/JOB_alternating_new_3k_10ep_{step}step/", "cuda:2",
    #                         30, f"selected_plans_3k_{step}_30.json")
    #     best_plan_prediction2("./training_data/JOB/", True,
    #                          f"./checkpoints/JOB_alternating_new_3k_10ep_{step}step/", "cuda:2",
    #                          f"selected_plans_3k_{step}_30.json", f"./results/JOB/RankPQO_3k_{step}_30.txt")
