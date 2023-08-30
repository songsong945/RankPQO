import psycopg2
import json
import os
import configure


# 1. Connect to PostgreSQL
def connect_to_pg():
    connection = psycopg2.connect(
        dbname=configure.dbname,
        user=configure.user,
        password=configure.password,
        host=configure.host,
        port=configure.port
    )
    return connection


# 2. Bind parameters and fetch execution plan
def fetch_execution_plan(connection, template, parameters):
    cursor = connection.cursor()

    cursor.execute("SET max_parallel_workers_per_gather TO 0;")

    query = cursor.mogrify(template, parameters).decode()
    cursor.execute(f"EXPLAIN (FORMAT JSON) {query}")
    plan = cursor.fetchone()
    cursor.close()
    return plan[0][0]


# 3. Iterate over meta_data and parameters to get all execution plans
def generate_plans_for_query(meta_data_path, parameter_path):
    # Load meta data and parameters
    with open(meta_data_path, 'r') as f:
        meta_data = json.load(f)
    with open(parameter_path, 'r') as f:
        parameters_list = json.load(f)

    connection = connect_to_pg()
    plans = {}
    for idx, params in enumerate(parameters_list.values()):
        # For each parameter set, fetch the execution plan
        plan = fetch_execution_plan(connection, meta_data['template'], params)
        plans[f"plan {idx + 1}"] = plan

    connection.close()
    return plans


# 4. Save execution plans as JSON
def save_execution_plans_for_all(data_directory):
    for subdir, _, _ in os.walk(data_directory):
        meta_data_path = os.path.join(subdir, "meta_data.json")
        parameter_path = os.path.join(subdir, "parameter.json")

        if os.path.isfile(meta_data_path) and os.path.isfile(parameter_path):
            print(f"Processing: {meta_data_path}")
            plans = generate_plans_for_query(meta_data_path, parameter_path)

            with open(os.path.join(subdir, "all_plans.json"), 'w') as f:
                json.dump(plans, f, indent=4)


if __name__ == "__main__":
    meta_data_path = '../training_data/JOB/'
    save_execution_plans_for_all(meta_data_path)
