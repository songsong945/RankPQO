import json
import psycopg2
import re

from decimal import Decimal

import configure
import os

def decimal_default(obj):
    if isinstance(obj, Decimal):
        return str(obj)  # 或者使用 float(obj) 如果不需要保留精确性
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")



def add_statistics(meta_data_path):
    connection = psycopg2.connect(
        host=configure.host,
        port=configure.port,
        dbname=configure.dbname3,
        user=configure.user,
        password=configure.password
    )
    cursor = connection.cursor()

    # 读取meta_data
    with open(meta_data_path, 'r') as file:
        meta_data = json.load(file)

    # 解析template字段以获取表名
    # tables_section = re.search(r"FROM(.*?)WHERE", meta_data["template"], re.S).group(1).strip()
    table_mapping = configure.mapping_tpcds
    # for table_definition in tables_section.split(","):
    #     table_name, alias = [item.strip() for item in table_definition.split(" AS ")]
    #     table_mapping[alias] = table_name

    # 为每个谓词获取统计信息
    for predicate in meta_data["predicates"]:
        table_column = predicate["column"]
        if '.' in table_column:
        # alias, column_name = table_column.split(".")
        # actual_table_name = table_mapping[alias]
            alias, column_name = table_column.split(".")
        else:
            column_name = table_column
        actual_table_name = table_mapping[table_column]


        # 对于 int 类型
        if predicate["data_type"] == "int":
            cursor.execute(
                f"SELECT MIN({column_name}), MAX({column_name}), COUNT(DISTINCT {column_name}) FROM (SELECT DISTINCT {column_name} FROM {actual_table_name}) sub")
            min_val, max_val, distinct_count = cursor.fetchone()
            predicate["min"] = min_val
            predicate["max"] = max_val
            predicate["max_len"] = distinct_count + 5

        # 对于 float 类型
        elif predicate["data_type"] == "float":
            cursor.execute(f"SELECT AVG({column_name}), VAR_SAMP({column_name}) FROM {actual_table_name}")
            mean, variance = cursor.fetchone()
            predicate["mean"] = mean
            predicate["variance"] = variance

        # 对于 text 类型
        elif predicate["data_type"] == "text":
            cursor.execute(
                f"SELECT ARRAY_AGG(DISTINCT {column_name}), COUNT(DISTINCT {column_name}) FROM {actual_table_name}")
            distinct_values, distinct_count = cursor.fetchone()
            predicate["distinct_values"] = distinct_values
            predicate["max_len"] = distinct_count + 5

    # 关闭数据库连接
    cursor.close()
    connection.close()

    # 更新meta_data.json文件
    with open(meta_data_path, 'w') as file:
        json.dump(meta_data, file, indent=4, default=decimal_default)


def process_all_meta_data(data_directory):
    for subdir, _, _ in os.walk(data_directory):
        meta_data_path = os.path.join(subdir, "meta_data.json")

        # 检查meta_data.json是否在子目录中
        if os.path.isfile(meta_data_path):
            print(f"Processing: {meta_data_path}")
            add_statistics(meta_data_path)


if __name__ == "__main__":
    meta_data_path = '../training_data/TPCDS/'
    process_all_meta_data(meta_data_path)
