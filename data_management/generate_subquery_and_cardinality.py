import json
import os
import random
import re
import time
from multiprocessing import Pool

import psycopg2
import sqlparse
from itertools import permutations, combinations

import configure


def connect_to_pg():
    connection = psycopg2.connect(
        dbname=configure.dbname,
        user=configure.user,
        password=configure.password,
        host=configure.host,
        port=configure.port
    )
    return connection

def extract_tables_and_predicates(sql):
    parsed = sqlparse.parse(sql)[0]
    tables = {}  # {alias: {'real_name': real_name, 'conditions': []}}
    where_clause = ""
    from_seen = False

    # 检查 token 是否为关键字 'FROM' 或 'WHERE'
    def is_keyword(token, keyword):
        return token.ttype is sqlparse.tokens.Keyword and str(token).strip().upper() == keyword

    for token in parsed.tokens:
        if from_seen:
            if isinstance(token, sqlparse.sql.IdentifierList):
                for identifier in token.get_identifiers():
                    real_name = identifier.get_real_name()
                    alias = identifier.get_alias() or real_name
                    tables[alias] = {'real_name': real_name, 'conditions': []}
            elif isinstance(token, sqlparse.sql.Identifier):
                real_name = token.get_real_name()
                alias = token.get_alias() or real_name
                tables[alias] = {'real_name': real_name, 'conditions': []}
        # 找到 WHERE 关键字后，解析整个 WHERE 子句
        if "WHERE" in str(token).upper():
            where_clause = str(token.parent).split('WHERE', 1)[1].strip()
            where_clause = where_clause.rstrip(';')

            # 使用改进的关键字检查
        if is_keyword(token, "FROM"):
            from_seen = True

    if where_clause:
        split_conditions = re.split(r'\bAND\b', where_clause, flags=re.IGNORECASE)
        final_conditions = []
        skip_next = False

        for i, condition in enumerate(split_conditions):
            condition = condition.strip()
            if skip_next:
                skip_next = False
                continue

            # 检查是否包含 BETWEEN，如果包含则将下一个条件合并
            if 'BETWEEN' in condition.upper():
                if i + 1 < len(split_conditions):
                    condition = condition + ' AND ' + split_conditions[i + 1].strip()
                    skip_next = True  # 跳过下一个条件，因为它已被合并
            final_conditions.append(condition)

        for alias in tables.keys():
            conditions = []
            # 使用正则表达式查找与当前别名相关的条件
            pattern = re.compile(r'(?<!\w){}\.'.format(re.escape(alias)))  # 确保别名作为整个词匹配
            for condition in final_conditions:
                if pattern.search(condition):
                    conditions.append(condition)
            tables[alias]['conditions'] = conditions

    return tables

def get_cardinality(connection, sql_query, path, param_values):
    cursor = connection.cursor()

    try:
        # 创建目录，如果目录不存在
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 使用 mogrify 格式化 SQL 查询
        sql_query = cursor.mogrify(sql_query, param_values).decode()

        # 打开文件进行写入
        with open(path, 'w') as file:
            tables = extract_tables_and_predicates(sql_query)
            results_cache = {}

            for i in range(2, len(tables) + 1):
                combos = combinations(tables.keys(), i)
                for combo in combos:
                    # 对表名进行排序，确保相同的组合只处理一次
                    sorted_combo = tuple(sorted(combo))

                    if sorted_combo not in results_cache:
                        from_clause = ', '.join([f"{tables[alias]['real_name']} AS {alias}" for alias in sorted_combo])
                        where_clauses = []
                        included_tables = set(sorted_combo)

                        for alias in sorted_combo:
                            for condition in tables[alias]['conditions']:
                                # 检查条件中的表是否在当前组合中
                                condition_tables = set(
                                    token.strip().split('.')[0] for token in condition.split() if '.' in token)
                                if condition_tables.issubset(included_tables):
                                    where_clauses.append(condition)

                        where_clause = ' AND '.join(where_clauses)

                        if where_clause:
                            query = f"SELECT COUNT(*) FROM {from_clause} WHERE {where_clause}"
                        else:
                            query = f"SELECT COUNT(*) FROM {from_clause}"

                        try:
                            cursor.execute(query)
                            count = cursor.fetchone()[0]
                            results_cache[sorted_combo] = count
                            # print(f"Query successful for: {', '.join(sorted_combo)}, Count: {count}")
                        except psycopg2.Error as e:
                            print(f"Query failed for: {', '.join(sorted_combo)}. Error: {str(e)}")
                            results_cache[sorted_combo] = None
                            connection.rollback()
                            cursor = connection.cursor()  # 重新获取游标

                        count = results_cache.get(sorted_combo)
                        if count is not None:
                            file.write(f"{','.join(sorted_combo)},:{count}\n")
    except Exception as e:
        # 捕获所有其他异常，打印错误信息
        print(f"An error occurred: {str(e)}")
        connection.rollback()  # 确保回滚事务
def evaluate_directory(subdir):
    connection = connect_to_pg()
    random.seed(42)

    with open(os.path.join(subdir, "meta_data.json"), 'r') as f_meta:
        meta_data = json.load(f_meta)

    with open(os.path.join(subdir, "parameter_new.json"), 'r') as f_params:
        parameters = json.load(f_params)

    print(f"Processing {subdir}...")

    template = meta_data["template"]
    param_keys = random.sample(list(parameters.keys()), min(200, len(parameters.keys())))
    param_keys = param_keys[160:]

    for param_key in param_keys:
        param_values = parameters[param_key]
        query = template.format(*param_values)

        get_cardinality(connection, query, f"/mnt/newpart/postgres/JOB/{os.path.basename(subdir)}/{param_key}.txt", param_values)

    print(f"Finished {subdir}...")

    connection.close()


def evaluate_all_mutil_process(data_directory):
    directories_to_process = []

    for subdir, _, files in os.walk(data_directory):
        if "meta_data.json" in files:
            directories_to_process.append(subdir)

    with Pool(processes=12) as pool:
        pool.map(evaluate_directory, directories_to_process)


if __name__ == "__main__":
    meta_data_path = '../training_data/JOB/'
    start_time = time.time()
    evaluate_all_mutil_process(meta_data_path)
    # evaluate_directory(meta_data_path)
    end_time = time.time()
    print(end_time - start_time)
