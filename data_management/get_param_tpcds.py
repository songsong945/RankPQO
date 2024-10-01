import os
import json
import re

def read_template_from_json(directory):
    """
    从给定目录中的 meta_data.json 文件读取 SQL 模板。
    """
    json_path = os.path.join(directory, 'meta_data.json')
    try:
        with open(json_path, 'r') as file:
            data = json.load(file)
            return data['template']
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return None


def extract_parameters_by_line(template, query):
    template_lines = template.strip().split('\n')
    query_lines = query.strip().split('\n')
    extracted_params = []

    for t_line, q_line in zip(template_lines, query_lines):
        if '%s' in t_line:
            t_line = re.sub(r'\s*\(\(|\)\)\s*', ' ', t_line).strip()
            q_line = re.sub(r'\s*\(\(|\)\)\s*', ' ', q_line).strip()
            if "cast" in t_line and "interval" in t_line:
                date_pattern = r"'\d{4}-\d{2}-\d{2}'"
                match = re.findall(date_pattern, q_line)
                if match:
                    extracted_params.extend([date.strip("'") for date in match])
            elif "ib_lower_bound" in t_line:
                match = q_line.split(">=")[1].split(' ')
                if match:
                    extracted_params.append(match[2])
            elif "ib_upper_bound" in t_line:
                match = q_line.split("<=")[1].split(' ')
                if match:
                    extracted_params.append(match[2])
            elif "in " in t_line or "IN " in t_line:
                pattern = r"IN \(([^)]+)\)"
                if "in" in t_line:
                    pattern = r"in \(([^)]+)\)"
                match = re.search(pattern, q_line)
                if match:
                    list_items = match.group(1).replace("'", "").split(',')
                    extracted_params.extend([item.strip() for item in list_items])
            elif "between" in t_line or "BETWEEN" in t_line:
                pattern = t_line.replace(r'%s', r'(\d+)')
                if "+ 23" in t_line:
                    pattern = pattern.replace("+ 23", r"\+ 23")
                elif "+ 2" in t_line:
                    pattern = pattern.replace("+ 2", r"\+ 2")
                elif "+ 1" in t_line:
                    pattern = pattern.replace("+ 1", r"\+ 1")
                elif "*" in t_line:
                    pattern = pattern.replace("*", "")
                    q_line = q_line.replace("*", "")
                match = re.search(pattern, q_line)
                if match:
                    extracted_params.extend(match.groups())
            else:
                pattern = t_line.replace(r'%s', r"'?([^']*)'?")
                match = re.search(pattern, q_line)
                if match:
                    extracted_params.extend(match.groups())

    # print(extracted_params)
    return extracted_params

def extract_parameters(template, query_directory):
    """
    从指定目录的查询文件中提取对应模板的参数。
    """
    parameters = {}
    template = '\n' + template
    for i in range(0, 50):  # 假设文件名格式为 query1.sql, query2.sql, ..., query50.sql
        query_path = os.path.join(query_directory, f'query_{i}.sql')
        try:
            with open(query_path, 'r') as file:
                query = file.read()
                parameters[f'parameter {i+1}'] = extract_parameters_by_line(template, query)
        except Exception as e:
            print(f"Error processing {query_path}: {e}")

    return parameters

def process_directories(base_dir, query_dir):
    """
    处理所有子目录以提取参数，并保存到 parameters.json 文件中。
    """
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        print(subdir)
        # if subdir != 'query100':
        #     continue
        if os.path.isdir(subdir_path):
            template = read_template_from_json(subdir_path)
            print(template.count('%s'))
            if template:
                # 对应的查询目录
                query_subdir_path = os.path.join(query_dir, subdir)
                if os.path.exists(query_subdir_path):
                    parameters = extract_parameters(template, query_subdir_path)
                    print(len(parameters[f'parameter 1']))
                    print(parameters[f'parameter 1'])
                    output_path = os.path.join(subdir_path, 'parameters.json')
                    with open(output_path, 'w') as file:
                        json.dump(parameters, file, indent=4)
                else:
                    print(f"No corresponding query directory found for {subdir_path}")

# 示例用法
base_dir = '../training_data/TPCDS'  # 基础目录，包含 meta_data.json 文件的子目录
query_dir = '../training_data/tpc-ds'  # 查询目录，包含 50 个查询文件的子目录
process_directories(base_dir, query_dir)
