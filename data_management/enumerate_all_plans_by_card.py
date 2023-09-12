import psycopg2
import json
import os
import configure

def generate_hints_from_plan(plan, alias):
    # Recursively search the plan for a specific alias and return its estimated rows
    def search_plan_for_alias(node, alias):
        if "Relation Name" in node and node["Alias"] == alias:
            return node["Plan Rows"]
        if "Plans" in node:
            for child in node["Plans"]:
                rows = search_plan_for_alias(child, alias)
                if rows:
                    return rows
        return None

    rows = search_plan_for_alias(plan["Plan"], alias)
    if rows:
        double_rows_hint = f"/*+ Rows({rows * 2}) */"
        half_rows_hint = f"/*+ Rows({rows / 2}) */"
        return [double_rows_hint, half_rows_hint]
    else:
        return []


def connect_to_pg():
    connection = psycopg2.connect(
        dbname=configure.dbname,
        user=configure.user,
        password=configure.password,
        host=configure.host,
        port=configure.port
    )
    return connection


def fetch_execution_plan(connection, template, parameters):
    cursor = connection.cursor()

    cursor.execute("SET max_parallel_workers_per_gather TO 0;")

    query = cursor.mogrify(template, parameters).decode()
    cursor.execute(f"EXPLAIN (FORMAT JSON) {query}")
    plan = cursor.fetchone()
    cursor.close()
    return plan[0][0]


def generate_plans_for_query(meta_data_path, parameter_path):
    with open(meta_data_path, 'r') as f:
        meta_data = json.load(f)
    with open(parameter_path, 'r') as f:
        parameters_list = json.load(f)

    connection = connect_to_pg()
    plans = {}

    table_aliases = [predicate["alias"] for predicate in meta_data["predicates"]]

    idx = 0
    for params in enumerate(parameters_list.values()):

        plan = fetch_execution_plan(connection, meta_data['template'], params)
        idx += 1
        plans[f"plan {idx}"] = plan

        for alias in table_aliases:
            hints = generate_hints_from_plan(plan, alias)
            for hint in hints:
                idx += 1
                modified_plan_with_hint = fetch_execution_plan(connection, hint+" "+meta_data['template'], params)
                plans[f"modified {idx}"] = modified_plan_with_hint

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
