import os
import json


def evaluate_directory(subdir):
    print(f"Processing {subdir}...")
    min_time = 0
    count_time_out = 0
    total_plans = 0

    with open(os.path.join(subdir, f"cost_matrix_all.json"), 'r') as f_costs:
        data = json.load(f_costs)

        for parameter, plans in data.items():
            # Get the minimum time from the plan values
            min_time += min(plans.values())

            count_time_out += sum(1 for time in plans.values() if time >= 3)

            total_plans += len(plans)

    print(f"Finished {subdir}...")
    return min_time, count_time_out, total_plans


def evaluate_all(data_directory):
    min_time_total = 0
    count_time_out_total = 0
    total_plans_total = 0

    for subdir, _, files in os.walk(data_directory):
        if f"cost_matrix_all.json" in files:
            min_time, count_time_out, total_plans = evaluate_directory(subdir)
            min_time_total += min_time
            count_time_out_total += count_time_out
            total_plans_total += total_plans

    print(f"Total minimum time: {min_time_total}")
    print(f"Total count of plans exceeding time threshold: {count_time_out_total}")
    print(f"Total number of plans: {total_plans_total}")


if __name__ == "__main__":
    meta_data_path = '../training_data/JOB/'
    evaluate_all(meta_data_path)
