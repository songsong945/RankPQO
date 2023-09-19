import json
import os
import glob

def _meta_path(base):
    return os.path.join(base, "meta_data.json")

def populate_column_op(dataset_path):
    templates = glob.glob(dataset_path+'*/')
    distinct_columns = set()
    distinct_op = set()
    for template in templates:
        with open(_meta_path(template), 'r') as f:
            meta = json.load(f)
            for pred in meta['predicates']:
                distinct_columns.add(pred['column'])
                op = pred['operator']
                if op=='like':
                    op = '~~'
                if op =='not like':
                    op = '!~~'
                distinct_op.add(op)
    return {
    'columns': list(distinct_columns),
    'operators': list(distinct_op)
    }


if __name__ == "__main__":

    dataset_path = '../training_data/JOB/'
    out = populate_column_op(dataset_path)
    with open(dataset_path+'columns_and_operators.json', 'w') as file:
        json.dump(out, file, indent=4)

