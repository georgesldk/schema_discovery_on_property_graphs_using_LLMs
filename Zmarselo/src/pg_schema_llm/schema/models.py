def compare_properties(gt_props, inf_props):
    gt_set = set(p["name"] for p in gt_props)
    inf_set = set(p["name"] for p in inf_props)

    matches = len(gt_set & inf_set)
    total = len(gt_set)

    return matches, total
