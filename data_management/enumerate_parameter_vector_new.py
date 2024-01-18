import logging
import random
import time

import numpy as np
import json
import os
import configure
import psycopg2

random.seed(1234)


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s')

def fetch_actual_latency(connection, query_with_hint, parameters):
    cursor = connection.cursor()
    cursor.execute("SET max_parallel_workers_per_gather TO 0;")

    query_with_hint = cursor.mogrify(query_with_hint, parameters).decode()

    print(query_with_hint)

    start_time = time.time()
    cursor.execute(query_with_hint)
    end_time = time.time()

    cursor.close()
    latency = end_time - start_time
    return latency


def connect_to_pg():
    connection = psycopg2.connect(
        dbname=configure.dbname,
        user=configure.user,
        password=configure.password,
        host=configure.host,
        port=configure.port
    )
    return connection


def sample_int(min_val, max_val):
    return str(random.randint(min_val, max_val))


def sample_float(mean, variance):
    return str(round(np.random.normal(mean, np.sqrt(variance)), 2))


def sample_text(distinct_values):
    non_empty_values = [value for value in distinct_values if value is not None and value != ""]
    return random.choice(non_empty_values) if non_empty_values else None


def generate_parameters(meta_data_path, num_samples=1000):
    with open(meta_data_path, 'r') as f:
        data = json.load(f)

    predicates = data["predicates"]
    template = data["template"]
    param_combinations = {}

    connection = connect_to_pg()

    for i in range(num_samples):
        j = 0
        while True:
            j = j+1
            params_for_sample = []
            for predicate in predicates:
                data_type = predicate["data_type"]
                preprocess_type = predicate["preprocess_type"]
                if data_type == "int" and preprocess_type == "one_hot":
                    params_for_sample.append(sample_int(predicate["min"], predicate["max"]))
                elif data_type == "float" and preprocess_type == "std_normalization":
                    params_for_sample.append(sample_float(predicate["mean"], predicate["variance"]))
                elif data_type == "text" and preprocess_type == "embedding":
                    params_for_sample.append(sample_text(predicate["distinct_values"]))

            print(params_for_sample)

            time = fetch_actual_latency(connection, template, params_for_sample)
            #logging.info(f"Executing time: {time}s")

            if time > 7:
                break

            #if time > 0.18:
            #    break

        logging.info(f"Sampling {j} times for {i}th vector")
        logging.info(f"Executing time: {time}s")

        param_combinations[f"parameter {i + 1}"] = params_for_sample

    return param_combinations


def enumerate_parameters_for_meta_data(meta_data_path, output_path, num_samples=1000):
    param_combinations = generate_parameters(meta_data_path, num_samples)
    with open(output_path, 'w') as f:
        json.dump(param_combinations, f, indent=4)


def generate_parameters_for_all(data_directory):
    subdir = os.path.join(data_directory, "20a")
    meta_data_path = os.path.join(subdir, "meta_data.json")
    parameter_output_path = os.path.join(subdir, "parameter_new.json")

    # 检查meta_data.json是否在子目录中
    if os.path.isfile(meta_data_path):
        # 输出日志
        print(f"Processing parameters for query at {subdir}")
        enumerate_parameters_for_meta_data(meta_data_path, parameter_output_path)



if __name__ == "__main__":
    meta_data_path = '../training_data/JOB/'
    generate_parameters_for_all(meta_data_path)
