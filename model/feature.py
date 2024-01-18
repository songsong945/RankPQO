import json
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURE_LIST = ['Node Type', 'Startup Cost',
                'Total Cost', 'Plan Rows', 'Plan Width']
LABEL_LIST = ['Actual Startup Time', 'Actual Total Time', 'Actual Self Time']

UNKNOWN_OP_TYPE = "Unknown"
SCAN_TYPES = ["Seq Scan", "Index Scan", "Index Only Scan", 'Bitmap Heap Scan']
JOIN_TYPES = ["Nested Loop", "Hash Join", "Merge Join"]
OTHER_TYPES = ['Bitmap Index Scan']
OP_TYPES = [UNKNOWN_OP_TYPE, "Hash", "Materialize", "Sort", "Aggregate", "Incremental Sort", "Limit"] \
    + SCAN_TYPES + JOIN_TYPES + OTHER_TYPES

OPERATORS = ['!~~', '~~', '<=', '>=', '=', '<', '>']
UNKNOWN_COLUMN = 'Unknown'
COLUMNS = [UNKNOWN_COLUMN,
    "t2.production_year","pi.note","cn.name","n.gender","it.info","t.title","t.episode_nr","it2.info","n.name","cct2.kind","it3.info","mi.note","mc.note","mi_idx.info","t.production_year","n.name_pcode_cf","kt.kind","n1.name","lt.link","it1.info","cn1.country_code","rt.role","mi_idx2.info","k.keyword","an.name","ct.kind","ci.note","chn.name","mi.info","cn.country_code","cct1.kind"
]

import re
def extract_predicates_from_condition(condition, alias):
    operators = OPERATORS
    predicates = []
    for op in operators:
        if op in condition:
            parts_list = [condition] if ' AND ' not in condition and ' OR ' not in condition else re.split(r'\s+AND\s+|\s+OR\s+', condition)
            for part in parts_list:
                if op in part:
                    components = re.split(re.escape(op), part)
                    left_col = clean_column_name(components[0].strip())
                    right_col = clean_column_name(components[1].strip())
                    full_left_col = alias + '.' + left_col
                    
                    # Check if both left and right are column references; if so, skip
                    if re.match(r'^[a-z_]+\.[a-z_]+$', left_col, re.I) and re.match(r'^[a-z_]+\.[a-z_]+$', right_col, re.I):
                        continue
                    
                    predicates.append((full_left_col, op))
    return predicates


def clean_column_name(raw_column):
    cleaned = raw_column
    # Removing type casting, such as ::text
    cleaned = re.sub(r'::[a-zA-Z_]+', '', cleaned)
    # Removing extra parentheses and trailing or leading punctuation
    cleaned = re.sub(r'^\W+|\W+$', '', cleaned)
    return cleaned

def extract_predicates_from_node(plan):
    predicates = []
    alias = plan.get('Alias', '')
    
    # List of keys which can have conditions
    relevant_keys = ['Join Filter', 'Hash Cond', 'Filter', 'Index Cond']

    for key in relevant_keys:
        if key in plan:
            predicates += extract_predicates_from_condition(plan[key], alias)

    return predicates

def json_str_to_json_obj(json_data):
    json_obj = json.loads(json_data)
    if type(json_obj) == list:
        assert len(json_obj) == 1
        json_obj = json_obj[0]
        assert type(json_obj) == dict
    return json_obj



def preprocess_z(vector: list,
                               params: list,
                               preprocessing_infos: list) -> torch.Tensor:
    """Generates a preprocessed vector for a given parameter input.

    input example
    vector = torch.tensor([25, 5.0, "apple"])
    params = [
        {"data_type": "int", "min": 0, "max": 100},
        {"data_type": "float"},
        {"data_type": "text", "distinct_values": ["apple", "banana", "cherry"]}
    ]
    preprocessing_infos = [
        {"type": "one_hot", "max_len" : 50},
        {"type": "std_normalization", "mean": 0.0, "variance": 1.0},
        {"type": "embedding", "output_dim": 5, "max_len" : 50}
    ]
    
    output example
    np.array([25, 0.6, 0])

    """

    processed_components = []
    vector = list(zip(*vector))
    for i, (param, preprocessing_info) in enumerate(zip(params, preprocessing_infos)):
        data_type = param["data_type"]
        preprocessing_type = preprocessing_info["type"]
        layer = vector[i]
        if data_type == "float" and preprocessing_type == "std_normalization":
            mean = preprocessing_info["mean"]
            std = torch.sqrt(preprocessing_info["variance"])
            processed_components.append((np.array(layer).astype(int) - mean) / std)
        elif data_type == "int":
            # shifted_layer = np.array(layer).astype(int) - param["min"]
            # processed_components.append(shifted_layer)
            if preprocessing_type == "embedding":
                vocab = {word: idx for idx, word in enumerate(param["distinct_values"])}
                num_oov_indices = preprocessing_info.get("num_oov_indices", 0)
                lookup_layer = np.array([vocab.get(la, len(vocab)) for la in layer])
                processed_components.append(lookup_layer)
            # elif preprocessing_type == "one_hot":
            #     processed_components.append(
            #         F.one_hot(shifted_layer.long(), num_classes=param["max"] - param["min"] + 1).float())

        elif data_type == "text":
            vocab = {word: idx for idx, word in enumerate(param["distinct_values"])}
            num_oov_indices = preprocessing_info.get("num_oov_indices", 0)
            lookup_layer = np.array([vocab.get(la, len(vocab)) for la in layer])
            processed_components.append(lookup_layer)
            # lookup_layer = torch.tensor(vocab.get(layer.item(), len(vocab)))
            # if preprocessing_type == "embedding":
            #     embed = nn.Embedding(len(vocab) + num_oov_indices,
            #                          preprocessing_info["output_dim"])
            #     processed_components.append(embed(lookup_layer))
            # elif preprocessing_type == "one_hot":
            #     processed_components.append(F.one_hot(lookup_layer, num_classes=len(vocab) + num_oov_indices).float())
        else:
            raise ValueError(f"Unsupported preprocessing: parameter type: {data_type}"
                             f" preprocessing type: {preprocessing_type}")



    ## Modified by zy, return the index of embedding/one-hot
    ## instead of the full vector  

    # Concatenate all processed components into a single vector
    return np.transpose(np.array(processed_components))
    # return torch.cat(processed_components, dim=-1)


class FeatureGenerator():

    def __init__(self, use_est = True, use_pred = False) -> None:
        ## assume either use ests or pred, but not both

        self.normalizer = None
        self.feature_parser = None

        self.use_est = use_est
        self.use_pred = use_pred
        self.columns = COLUMNS




    def fit(self, trees):
        exec_times = []
        startup_costs = []
        total_costs = []
        rows = []
        input_relations = set()
        rel_type = set()

        def recurse(n):
            startup_costs.append(n["Startup Cost"])
            total_costs.append(n["Total Cost"])
            rows.append(n["Plan Rows"])
            rel_type.add(n["Node Type"])
            if "Relation Name" in n:
                # base table
                input_relations.add(n["Relation Name"])

            if "Plans" in n:
                for child in n["Plans"]:
                    recurse(child)

        for tree in trees:
            # json_obj = json_str_to_json_obj(tree)
            json_obj = tree
            if "Execution Time" in json_obj:
                exec_times.append(float(json_obj["Execution Time"]))
            recurse(json_obj["Plan"])

        startup_costs = np.array(startup_costs)
        total_costs = np.array(total_costs)
        rows = np.array(rows)

        startup_costs = np.log(startup_costs + 1)
        total_costs = np.log(total_costs + 1)
        rows = np.log(rows + 1)

        startup_costs_min = np.min(startup_costs)
        startup_costs_max = np.max(startup_costs)
        total_costs_min = np.min(total_costs)
        total_costs_max = np.max(total_costs)
        rows_min = np.min(rows)
        rows_max = np.max(rows)

        #print("RelType : ", rel_type)

        if len(exec_times) > 0:
            exec_times = np.array(exec_times)
            exec_times = np.log(exec_times + 1)
            exec_times_min = np.min(exec_times)
            exec_times_max = np.max(exec_times)
            self.normalizer = Normalizer(
                {"Execution Time": exec_times_min, "Startup Cost": startup_costs_min,
                 "Total Cost": total_costs_min, "Plan Rows": rows_min},
                {"Execution Time": exec_times_max, "Startup Cost": startup_costs_max,
                 "Total Cost": total_costs_max, "Plan Rows": rows_max})
        else:
            self.normalizer = Normalizer(
                {"Startup Cost": startup_costs_min,
                 "Total Cost": total_costs_min, "Plan Rows": rows_min},
                {"Startup Cost": startup_costs_max,
                 "Total Cost": total_costs_max, "Plan Rows": rows_max})
        self.feature_parser = AnalyzeJsonParser(self.normalizer, list(input_relations))

    def fit_pred_model(self, trees):
        exec_times = []
        startup_costs = []
        total_costs = []
        rows = []
        input_relations = set()
        rel_type = set()

        def recurse(n):
            startup_costs.append(n["Startup Cost"])
            total_costs.append(n["Total Cost"])
            rows.append(n["Plan Rows"])
            ## Record columns
            col_op_pairs = extract_predicates_from_node(n)
            for col,op in col_op_pairs:
                if col not in self.columns:
                    self.columns.append(col)

            rel_type.add(n["Node Type"])
            if "Relation Name" in n:
                # base table
                input_relations.add(n["Relation Name"])

            if "Plans" in n:
                for child in n["Plans"]:
                    recurse(child)

        for tree in trees:
            # json_obj = json_str_to_json_obj(tree)
            json_obj = tree
            if "Execution Time" in json_obj:
                exec_times.append(float(json_obj["Execution Time"]))
            recurse(json_obj["Plan"])

        startup_costs = np.array(startup_costs)
        total_costs = np.array(total_costs)
        rows = np.array(rows)

        startup_costs = np.log(startup_costs + 1)
        total_costs = np.log(total_costs + 1)
        rows = np.log(rows + 1)

        startup_costs_min = np.min(startup_costs)
        startup_costs_max = np.max(startup_costs)
        total_costs_min = np.min(total_costs)
        total_costs_max = np.max(total_costs)
        rows_min = np.min(rows)
        rows_max = np.max(rows)

        #print("RelType : ", rel_type)

        if len(exec_times) > 0:
            exec_times = np.array(exec_times)
            exec_times = np.log(exec_times + 1)
            exec_times_min = np.min(exec_times)
            exec_times_max = np.max(exec_times)
            self.normalizer = Normalizer(
                {"Execution Time": exec_times_min, "Startup Cost": startup_costs_min,
                 "Total Cost": total_costs_min, "Plan Rows": rows_min},
                {"Execution Time": exec_times_max, "Startup Cost": startup_costs_max,
                 "Total Cost": total_costs_max, "Plan Rows": rows_max})
        else:
            self.normalizer = Normalizer(
                {"Startup Cost": startup_costs_min,
                 "Total Cost": total_costs_min, "Plan Rows": rows_min},
                {"Startup Cost": startup_costs_max,
                 "Total Cost": total_costs_max, "Plan Rows": rows_max})
        self.feature_parser = AnalyzeJsonParser(self.normalizer, list(input_relations))

    # def transform(self, trees):
    #     local_features = []
    #     y = []
    #     for tree in trees:
    #         json_obj = json_str_to_json_obj(tree)
    #         if type(json_obj["Plan"]) != dict:
    #             json_obj["Plan"] = json.loads(json_obj["Plan"])
    #         local_feature = self.feature_parser.extract_feature(
    #             json_obj["Plan"])
    #         local_features.append(local_feature)
    #
    #         if "Execution Time" in json_obj:
    #             label = float(json_obj["Execution Time"])
    #             if self.normalizer.contains("Execution Time"):
    #                 label = self.normalizer.norm(label, "Execution Time")
    #             y.append(label)
    #         else:
    #             y.append(None)
    #     return local_features, y

    def transform(self, trees):
        local_features = []
        for tree in trees:
            # json_obj = json_str_to_json_obj(tree)
            json_obj = tree
            if type(json_obj["Plan"]) != dict:
                json_obj["Plan"] = json.loads(json_obj["Plan"])
            if self.use_est:
                local_feature = self.feature_parser.extract_feature(
                    json_obj["Plan"])
            else:
                assert self.use_pred
                local_feature = self.feature_parser.extract_pred_feature(
                    json_obj["Plan"])                
            local_features.append(local_feature)


        return local_features

    def transform_z(self, Z, params, preprocessing_infos):
        return preprocess_z(Z, params, preprocessing_infos)



class SampleEntity():
    def __init__(self, node_type: np.ndarray, startup_cost: float, total_cost: float,
                 rows: float, width: int,
                 left, right,
                 startup_time: float, total_time: float,
                 input_tables: list, encoded_input_tables: list,
                 predicates = [], pred_encoding = None) -> None:
        self.node_type = node_type
        self.startup_cost = startup_cost
        self.total_cost = total_cost
        self.rows = rows
        self.width = width
        self.left = left
        self.right = right
        self.startup_time = startup_time
        self.total_time = total_time
        self.input_tables = input_tables
        self.encoded_input_tables = encoded_input_tables

        
        self.use_est = True
        self.use_pred = False

        self.predicates = predicates
        self.pred_encoding = pred_encoding

        if self.pred_encoding[-1] == 0:
            self.pred_encoding[-1] = 1



    def __str__(self):
        return "{%s, %s, %s, %s, %s, [%s], [%s], %s, %s, [%s], [%s]}" % (self.node_type,
                                                                        self.startup_cost, self.total_cost, self.rows,
                                                                        self.width, self.left, self.right,
                                                                        self.startup_time, self.total_time,
                                                                        self.input_tables, self.encoded_input_tables)

    def get_feature(self):
        # return np.hstack((self.node_type, np.array([self.width, self.rows])))
        if self.use_est:
            return np.hstack((self.node_type, np.array(self.encoded_input_tables), np.array([self.width, self.rows])))
        else:
            assert(self.use_pred)
            return np.hstack((self.node_type, np.array(self.encoded_input_tables), self.pred_encoding))

    def get_feature_len(self):
        return len(self.get_feature())


    def get_left(self):
        return self.left

    def get_right(self):
        return self.right

    def subtrees(self):
        trees = []
        trees.append(self)
        if self.left is not None:
            trees += self.left.subtrees()
        if self.right is not None:
            trees += self.right.subtrees()
        return trees


class Normalizer():
    def __init__(self, mins: dict, maxs: dict) -> None:
        self._mins = mins
        self._maxs = maxs

    def norm(self, x, name):
        if name not in self._mins or name not in self._maxs:
            raise Exception("fail to normalize " + name)

        return (np.log(x + 1) - self._mins[name]) / (self._maxs[name] - self._mins[name])

    def inverse_norm(self, x, name):
        if name not in self._mins or name not in self._maxs:
            raise Exception("fail to inversely normalize " + name)

        return np.exp((x * (self._maxs[name] - self._mins[name])) + self._mins[name]) - 1

    def contains(self, name):
        return name in self._mins and name in self._maxs


class FeatureParser(metaclass=ABCMeta):

    @abstractmethod
    def extract_feature(self, json_data) -> SampleEntity:
        pass


# the json file is created by "EXPLAIN (ANALYZE, VERBOSE, COSTS, BUFFERS, TIMING, SUMMARY, FORMAT JSON) ..."
class AnalyzeJsonParser(FeatureParser):

    def __init__(self, normalizer: Normalizer, input_relations: list, columns = COLUMNS) -> None:
        self.normalizer = normalizer
        self.input_relations = input_relations
        self.max_predicate_len = 30
        self.columns = columns
        self.ops = OPERATORS

    def extract_feature(self, json_rel) -> SampleEntity:
        left = None
        right = None
        input_relations = []

        if 'Plans' in json_rel:
            children = json_rel['Plans']
            assert len(children) <= 2 and len(children) > 0
            left = self.extract_feature(children[0])
            input_relations += left.input_tables

            if len(children) == 2:
                right = self.extract_feature(children[1])
                input_relations += right.input_tables
            else:
                right = SampleEntity(op_to_one_hot(UNKNOWN_OP_TYPE), 0, 0, 0, 0,
                                     None, None, 0, 0, [], self.encode_relation_names([]))

        node_type = op_to_one_hot(json_rel['Node Type'])
        # startup_cost = self.normalizer.norm(float(json_rel['Startup Cost']), 'Startup Cost')
        # total_cost = self.normalizer.norm(float(json_rel['Total Cost']), 'Total Cost')
        startup_cost = None
        total_cost = None
        rows = self.normalizer.norm(float(json_rel['Plan Rows']), 'Plan Rows')
        width = int(json_rel['Plan Width'])

        if json_rel['Node Type'] in SCAN_TYPES:
            input_relations.append(json_rel["Relation Name"])

        startup_time = None
        if 'Actual Startup Time' in json_rel:
            startup_time = float(json_rel['Actual Startup Time'])
        total_time = None
        if 'Actual Total Time' in json_rel:
            total_time = float(json_rel['Actual Total Time'])


        return SampleEntity(node_type, startup_cost, total_cost, rows, width, left,
                            right, startup_time, total_time,
                            input_relations, self.encode_relation_names(input_relations))


    def extract_pred_feature(self, json_rel) -> SampleEntity:
        left = None
        right = None
        input_relations = []

        if 'Plans' in json_rel:
            children = json_rel['Plans']
            assert len(children) <= 2 and len(children) > 0
            left = self.extract_pred_feature(children[0])
            input_relations += left.input_tables

            if len(children) == 2:
                right = self.extract_pred_feature(children[1])
                input_relations += right.input_tables
            else:
                right = SampleEntity(op_to_one_hot(UNKNOWN_OP_TYPE), 0, 0, 0, 0,
                                     None, None, 0, 0, [], self.encode_relation_names([]),
                                     [],np.zeros(2 * self.max_predicate_len + 1, dtype=int))
                right.use_pred = True
                right.use_est = False

        node_type = op_to_one_hot(json_rel['Node Type'])
        # startup_cost = self.normalizer.norm(float(json_rel['Startup Cost']), 'Startup Cost')
        # total_cost = self.normalizer.norm(float(json_rel['Total Cost']), 'Total Cost')
        startup_cost = None
        total_cost = None
        rows = self.normalizer.norm(float(json_rel['Plan Rows']), 'Plan Rows')
        width = int(json_rel['Plan Width'])

        if json_rel['Node Type'] in SCAN_TYPES:
            input_relations.append(json_rel["Relation Name"])

        startup_time = None
        if 'Actual Startup Time' in json_rel:
            startup_time = float(json_rel['Actual Startup Time'])
        total_time = None
        if 'Actual Total Time' in json_rel:
            total_time = float(json_rel['Actual Total Time'])

        predicates = extract_predicates_from_node(json_rel)
        cols = []
        ops = []
        for col, op in predicates:
            cols.append(self.columns.index(col))
            ops.append(self.ops.index(op))

        encoded_array = np.zeros(2 * self.max_predicate_len + 1, dtype=int)

        encoded_array[:len(cols)] = cols
        encoded_array[self.max_predicate_len:self.max_predicate_len + len(ops)] = ops
        encoded_array[-1] = len(cols)

        se = SampleEntity(node_type, startup_cost, total_cost, rows, width, left,
                            right, startup_time, total_time,
                            input_relations, self.encode_relation_names(input_relations),
                            predicates,encoded_array)
        se.use_est = False
        se.use_pred = True
        return se




    def encode_relation_names(self, l):
        encode_arr = np.zeros(len(self.input_relations) + 1)

        for name in l:
            if name not in self.input_relations:
                # -1 means UNKNOWN
                encode_arr[-1] += 1
            else:
                encode_arr[list(self.input_relations).index(name)] += 1
        return encode_arr


def op_to_one_hot(op_name):
    arr = np.zeros(len(OP_TYPES))
    if op_name not in OP_TYPES:
        arr[OP_TYPES.index(UNKNOWN_OP_TYPE)] = 1
    else:
        arr[OP_TYPES.index(op_name)] = 1
    return arr
