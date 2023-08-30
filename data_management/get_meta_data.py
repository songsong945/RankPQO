import psycopg2
import json
import sqlparse
import os


def is_numeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def determine_data_type(value):
    if value[0] in ("'", '"', '`'):
        return "text"
    elif "." in value:
        return "float"
    else:
        return "int"


def determine_preprocess_type(data_type):
    if data_type == "text":
        return "embedding"
    elif data_type == "float":
        return "std_normalization"
    elif data_type == "int":
        return "one_hot"


def extract_conditions_from_token(token):
    conditions = []
    if 'BETWEEN' in token.value.upper() and 'AND' in token.value.upper():
        pass
    elif isinstance(token, sqlparse.sql.Parenthesis) or (isinstance(token, sqlparse.sql.Operation) and token.value.upper() in ['AND', 'OR']):
        for item in token.tokens:
            conditions.extend(extract_conditions_from_token(item))
    elif isinstance(token, sqlparse.sql.Comparison):
        left, operator, right = extract_comparison(token)
        if right and (right[0] in ("'", '"', '`') or is_numeric(right)):
            conditions.append({
                "column": left,
                "operator": operator,
                "value": right
            })
    return conditions


def extract_comparison(item):
    operators = ['=', '>', '<', '>=', '<=', '<>', '!=', 'LIKE', 'NOT LIKE']
    operator_token = None

    for token in item.tokens:
        if token.value.upper() in operators:
            operator_token = token
            break

    if not operator_token:
        return None, None, None

    left = "".join(tok.value for tok in item.tokens[:item.tokens.index(operator_token)])
    right = "".join(tok.value for tok in item.tokens[item.tokens.index(operator_token) + 1:])

    return left.strip(), operator_token.value.upper(), right.strip()


def generate_meta_data_from_sql(file_path, meta_data_path, template_id):
    with open(file_path, 'r') as file:
        sql = file.read()

    parsed = sqlparse.parse(sql)[0]

    where_conditions = []
    for token in parsed.tokens:
        if token.ttype is None and isinstance(token, sqlparse.sql.Where):
            for item in token.tokens:
                conditions = extract_conditions_from_token(item)
                where_conditions.extend(conditions)

    # Build template and predicates
    template = sql
    predicates = []

    for condition in where_conditions:
        column = condition["column"]
        alias, column_name = column.split(".")
        data_type = determine_data_type(condition["value"])
        preprocess_type = determine_preprocess_type(data_type)
        predicates.append({
            "alias": alias,
            "column": column,
            "operator": condition["operator"].lower(),
            "data_type": data_type,
            "preprocess_type": preprocess_type
        })
        template = template.replace(condition["value"], "%s", 1)

    template_json = {
        "template_id": template_id,
        "template": template,
        "predicates": predicates
    }
    with open(meta_data_path, 'w') as outfile:
        json.dump(template_json, outfile, indent=4)


def process_and_save_all_sql_files(source_directory, target_directory):
    # 列出目录中的所有文件
    files = [f for f in os.listdir(source_directory) if os.path.isfile(os.path.join(source_directory, f))]

    # 过滤出.sql文件
    sql_files = [file for file in files if file.endswith('.sql')]

    for sql_file in sql_files:
        source_file_path = os.path.join(source_directory, sql_file)
        query_id = sql_file.split('.')[0]

        # 为每个.sql文件创建一个同名目录
        target_folder_path = os.path.join(target_directory, query_id)
        if not os.path.exists(target_folder_path):
            os.makedirs(target_folder_path)

        # 生成meta_data并保存到目标目录
        target_file_path = os.path.join(target_folder_path, "meta_data.json")
        generate_meta_data_from_sql(source_file_path, target_file_path, query_id)

    print("Processing complete!")


if __name__ == "__main__":
    sql_file_path = '../training_data/join-order-benchmark/'
    meta_data_path = '../training_data/JOB/'
    process_and_save_all_sql_files(sql_file_path, meta_data_path)
