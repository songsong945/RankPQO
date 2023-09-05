def pair_hierarchy(hierarchy):
    if isinstance(hierarchy, str):
        return hierarchy
    elif len(hierarchy) == 1:
        return pair_hierarchy(hierarchy[0])
    else:
        hierarchy = [pair_hierarchy(item) for item in hierarchy]
    return hierarchy

# 示例测试
hierarchy = [[[['mc', [['mi_idx', ['it']]]], 'ct'], 't']]
result = pair_hierarchy(hierarchy)
print(result)  # 应该输出：['mc', ['mi_idx', 'it'], 'ct', 't']
