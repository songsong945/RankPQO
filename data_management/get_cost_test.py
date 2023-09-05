import unittest

from evaluate_cost_matrix import evaluate_directory


def fix_leading(input_str):
    # 先移除所有空格
    input_str = input_str.replace(' ', '')

    # 再添加空格在每个括号前后，这样我们可以通过空格来分割字符串
    input_str = input_str.replace('(', ' ( ')
    input_str = input_str.replace(')', ' ) ')

    # 使用空格分割字符串，形成一个单词列表
    words = input_str.split()

    # 初始化一个堆栈来跟踪括号
    stack = []
    removal_indices = []

    for idx, word in enumerate(words):
        # 如果找到左括号，将它和它的索引压入堆栈
        if word == '(':
            stack.append((word, idx))
        elif word == ')':
            # 如果找到右括号，检查堆栈顶部的项
            last_item, last_index = stack.pop()
            # 如果在这对括号之间只有一个单词，标记这两个括号以便后续删除
            if idx - last_index == 2:
                removal_indices.extend([last_index, idx])

    # 倒序移除标记的单词，因为从前往后删除会改变索引
    for index in sorted(removal_indices, reverse=True):
        del words[index]

    return ''.join(words)


# 测试
input_str = "Leading((((mc ((mi_idx (it)))) ct) t))"


def generate_hint_from_plan(plan):
    node = plan['Plan']
    hints = []

    def traverse_node(node):
        node_type = node['Node Type']
        rels = []  # Flattened list of relation names/aliases
        leading = []  # Hierarchical structure for LEADING hint

        # PG uses the former & the extension expects the latter.
        node_type = node_type.replace(' ', '')
        node_type = node_type.replace('NestedLoop', 'NestLoop')

        if 'Relation Name' in node:  # If it's a scan operation
            relation = node.get('Alias', node['Relation Name'])  # Prefer alias if exists
            if node_type in ['IndexScan', 'SeqScan']:
                hint = node_type + '(' + relation + ')'
                hints.append(hint)
            return [relation], relation
        else:
            if 'Plans' in node:
                for child in node['Plans']:
                    a, b = traverse_node(child)
                    rels.extend(a)
                    if b:  # Only add if it's not None
                        leading.append(b)
            if node_type in ['HashJoin', 'MergeJoin', 'NestLoop']:
                join_hint = node_type + '(' + ' '.join(rels) + ')'
                hints.append(join_hint)
            return rels, leading

    _, leading_hierarchy = traverse_node(node)

    # Ensure each parentheses in LEADING hint contains exactly two objects
    def pair_hierarchy(hierarchy):
        while len(hierarchy) > 2:
            hierarchy = [hierarchy[:2]] + hierarchy[2:]
        return hierarchy

    leading_hierarchy = pair_hierarchy(leading_hierarchy)

    leading = 'Leading' + str(leading_hierarchy) \
        .replace('\'', '') \
        .replace('[', '(') \
        .replace(']', ')') \
        .replace(',', '')
    hints.append(leading)

    query_hint = '\n '.join(hints)
    return query_hint


# Test the function
plan = {
        "Plan": {
            "Node Type": "Aggregate",
            "Strategy": "Plain",
            "Partial Mode": "Simple",
            "Parallel Aware": False,
            "Startup Cost": 83578.9,
            "Total Cost": 83578.91,
            "Plan Rows": 1,
            "Plan Width": 68,
            "Plans": [
                {
                    "Node Type": "Nested Loop",
                    "Parent Relationship": "Outer",
                    "Parallel Aware": False,
                    "Join Type": "Inner",
                    "Startup Cost": 25652.01,
                    "Total Cost": 83578.89,
                    "Plan Rows": 1,
                    "Plan Width": 45,
                    "Inner Unique": True,
                    "Join Filter": "(mc.movie_id = t.id)",
                    "Plans": [
                        {
                            "Node Type": "Nested Loop",
                            "Parent Relationship": "Outer",
                            "Parallel Aware": False,
                            "Join Type": "Inner",
                            "Startup Cost": 25651.58,
                            "Total Cost": 83578.3,
                            "Plan Rows": 1,
                            "Plan Width": 32,
                            "Inner Unique": True,
                            "Plans": [
                                {
                                    "Node Type": "Hash Join",
                                    "Parent Relationship": "Outer",
                                    "Parallel Aware": False,
                                    "Join Type": "Inner",
                                    "Startup Cost": 25651.45,
                                    "Total Cost": 83577.69,
                                    "Plan Rows": 2,
                                    "Plan Width": 36,
                                    "Inner Unique": False,
                                    "Hash Cond": "(mc.movie_id = mi_idx.movie_id)",
                                    "Plans": [
                                        {
                                            "Node Type": "Seq Scan",
                                            "Parent Relationship": "Outer",
                                            "Parallel Aware": False,
                                            "Relation Name": "movie_companies",
                                            "Alias": "mc",
                                            "Startup Cost": 0.0,
                                            "Total Cost": 57925.92,
                                            "Plan Rows": 81,
                                            "Plan Width": 32,
                                            "Filter": "(((note)::text !~~ '(2013) (Hungary) (TV) (re-release) (Story 4)'::text) AND ((note)::text ~~ '(2010) (Australia) (all media) (Wii version)'::text))"
                                        },
                                        {
                                            "Node Type": "Hash",
                                            "Parent Relationship": "Inner",
                                            "Parallel Aware": False,
                                            "Startup Cost": 25497.42,
                                            "Total Cost": 25497.42,
                                            "Plan Rows": 12322,
                                            "Plan Width": 4,
                                            "Plans": [
                                                {
                                                    "Node Type": "Hash Join",
                                                    "Parent Relationship": "Outer",
                                                    "Parallel Aware": False,
                                                    "Join Type": "Inner",
                                                    "Startup Cost": 2.41,
                                                    "Total Cost": 25497.42,
                                                    "Plan Rows": 12322,
                                                    "Plan Width": 4,
                                                    "Inner Unique": True,
                                                    "Hash Cond": "(mi_idx.info_type_id = it.id)",
                                                    "Plans": [
                                                        {
                                                            "Node Type": "Seq Scan",
                                                            "Parent Relationship": "Outer",
                                                            "Parallel Aware": False,
                                                            "Relation Name": "movie_info_idx",
                                                            "Alias": "mi_idx",
                                                            "Startup Cost": 0.0,
                                                            "Total Cost": 21735.34,
                                                            "Plan Rows": 1380034,
                                                            "Plan Width": 8
                                                        },
                                                        {
                                                            "Node Type": "Hash",
                                                            "Parent Relationship": "Inner",
                                                            "Parallel Aware": False,
                                                            "Startup Cost": 2.4,
                                                            "Total Cost": 2.4,
                                                            "Plan Rows": 1,
                                                            "Plan Width": 4,
                                                            "Plans": [
                                                                {
                                                                    "Node Type": "Seq Scan",
                                                                    "Parent Relationship": "Outer",
                                                                    "Parallel Aware": False,
                                                                    "Relation Name": "info_type",
                                                                    "Alias": "it",
                                                                    "Startup Cost": 0.0,
                                                                    "Total Cost": 2.4,
                                                                    "Plan Rows": 1,
                                                                    "Plan Width": 4,
                                                                    "Filter": "((info)::text = 'essays'::text)"
                                                                }
                                                            ]
                                                        }
                                                    ]
                                                }
                                            ]
                                        }
                                    ]
                                },
                                {
                                    "Node Type": "Index Scan",
                                    "Parent Relationship": "Inner",
                                    "Parallel Aware": False,
                                    "Scan Direction": "Forward",
                                    "Index Name": "company_type_pkey",
                                    "Relation Name": "company_type",
                                    "Alias": "ct",
                                    "Startup Cost": 0.13,
                                    "Total Cost": 0.3,
                                    "Plan Rows": 1,
                                    "Plan Width": 4,
                                    "Index Cond": "(id = mc.company_type_id)",
                                    "Filter": "((kind)::text = 'special effects companies'::text)"
                                }
                            ]
                        },
                        {
                            "Node Type": "Index Scan",
                            "Parent Relationship": "Inner",
                            "Parallel Aware": False,
                            "Scan Direction": "Forward",
                            "Index Name": "title_pkey",
                            "Relation Name": "title",
                            "Alias": "t",
                            "Startup Cost": 0.43,
                            "Total Cost": 0.58,
                            "Plan Rows": 1,
                            "Plan Width": 25,
                            "Index Cond": "(id = mi_idx.movie_id)",
                            "Filter": "(production_year > 1890)"
                        }
                    ]
                }
            ]
        }
    }


class MyTestCase(unittest.TestCase):
    meta_data_path = '../training_data/example_query/1c'

    def test_something(self):
        evaluate_directory(self.meta_data_path)
        # print(fix_leading(input_str))
        # print(generate_hint_from_plan(plan))

if __name__ == '__main__':
    unittest.main()
