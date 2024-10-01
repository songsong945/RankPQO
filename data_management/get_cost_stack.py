import json
import logging
import time

import psycopg2

import configure

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s')


def connect_to_pg():
    connection = psycopg2.connect(
        dbname=configure.dbname2,
        user=configure.user,
        password=configure.password,
        host=configure.host,
        port=configure.port
    )
    return connection


def fetch_actual_latency(connection, query_with_hint):
    cursor = connection.cursor()
    cursor.execute("SET max_parallel_workers_per_gather TO 0;")

    start_time = time.perf_counter()
    cursor.execute(query_with_hint)
    end_time = time.perf_counter()

    cursor.close()
    latency = end_time - start_time
    return latency


def extract_all_query(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    all_queries = {}
    for key, value in data.items():
        if isinstance(value, dict) and 'query' in value:
            query_details = {
                'query': value['query'],
                'predicates': value.get('predicates', []),
                'params': value.get('params', [])
            }
            all_queries[key] = query_details
        elif isinstance(value, dict):
            inner_queries = extract_all_query(value)
            all_queries.update(inner_queries)
    return all_queries


def run_queries(all_queries):
    connection = connect_to_pg()
    total = 0
    t_latency = 0
    for query_id, details in all_queries.items():
        print(f"Query ID: {query_id} and param nums: {len(details['params'])}")
        total += len(details['params'])
        for params in details['params']:
            formatted_query = details['query']
            for i, param in enumerate(params):
                formatted_query = formatted_query.replace(f'@param{i}', f"{param}")
            query = f"/*+ */ EXPLAIN ANALYZE " + formatted_query
            latency = fetch_actual_latency(connection, query)
            t_latency += latency
            logging.info(f"Query {query_id}: execute {latency}s")

    print(total)
    print(t_latency)


filepath = '../training_data/stack/stack_query_templates_with_metadata.json'
queries = extract_all_query(filepath)
run_queries(queries)
