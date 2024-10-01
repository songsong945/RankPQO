import json
import logging
import os
import random
import time
import psycopg2
import shutil
from data_management import configure


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


def connect_to_pg_job():
    connection = psycopg2.connect(
        dbname=configure.dbname,
        user=configure.user,
        password=configure.password,
        host=configure.host,
        port=configure.port
    )
    return connection


def PG_original(data):
    execution_time_total = 0.0

    all_folders = []
    for subdir, _, files in os.walk(data):
        if 'a' in os.path.basename(subdir) and "meta_data.json" in files:
            all_folders.append(os.path.basename(subdir))

    connection = connect_to_pg_job()

    for folder in all_folders:
        template_id = folder

        path = os.path.join(data, template_id)

        with open(os.path.join(path, "meta_data.json"), 'r') as f_meta:
            meta_data = json.load(f_meta)

        with open(os.path.join(path, "parameter_new.json"), 'r') as f_params:
            parameters = json.load(f_params)

        print(f"Processing {path}...")

        template = meta_data["template"]
        param_keys = random.sample(list(parameters.keys()), min(200, len(parameters.keys())))
        test_keys_list = param_keys[160:]

        for param_key in test_keys_list:
            parameter = parameters[param_key]

            query = f"/*+ */ " + template

            query = query.format(*parameter)

            latency = fetch_actual_latency(connection, query, parameter)

            logging.info(f"Query {folder}-{param_key}: execute {latency}s")

            execution_time_total += latency

    logging.info(f"total execution time: {execution_time_total}s")


def PG_original_true_card(data):
    execution_time_total = 0.0

    all_folders = []
    for subdir, _, files in os.walk(data):
        if 'a' in os.path.basename(subdir) and "meta_data.json" in files:
            all_folders.append(os.path.basename(subdir))

    connection = connect_to_pg_job()

    for folder in all_folders:
        template_id = folder

        path = os.path.join(data, template_id)

        with open(os.path.join(path, "meta_data.json"), 'r') as f_meta:
            meta_data = json.load(f_meta)

        with open(os.path.join(path, "parameter_new.json"), 'r') as f_params:
            parameters = json.load(f_params)

        print(f"Processing {path}...")

        template = meta_data["template"]
        param_keys = random.sample(list(parameters.keys()), min(200, len(parameters.keys())))
        test_keys_list = param_keys[160:]

        for param_key in test_keys_list:
            parameter = parameters[param_key]

            try:
                shutil.copyfile(f"/mnt/newpart/postgres/JOB/{template_id}/{param_key}.txt",
                                "/home/mosonsong/postgresql12/postgresql-12.5/info.txt")
            except FileNotFoundError:
                print(f"Source file {template_id}/{param_key}.txt does not exist")
            except PermissionError:
                print("Permission denied while copying the file")
            except Exception as e:
                print(f"An error occurred while copying the file: {e}")

            query = f"/*+ */ " + template

            query = query.format(*parameter)

            latency = fetch_actual_latency(connection, query, parameter)

            logging.info(f"Query {folder}-{param_key}: execute {latency}s")

            execution_time_total += latency

    logging.info(f"total execution time: {execution_time_total}s")


if __name__ == '__main__':
    PG_original("./training_data/JOB/")
    PG_original_true_card("./training_data/JOB/")
