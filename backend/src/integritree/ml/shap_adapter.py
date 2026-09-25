"""Prediction-equivalent RF representation for SHAP's interventional C kernel.

The kernel uses float32 thresholds and signed 16-bit node indexes (SHAP 0.52).
Flooring thresholds to the float32 grid preserves sklearn's predicates on its
float32 inputs. Large trees are expressed as a SUM of gated subtrees: each
component returns zero outside its original path. Linearity preserves the
interventional Shapley values, including the original reference distribution.
The trained estimator and its saved artifacts are never modified.
"""
import numpy as np


def forest_representation(model, max_nodes=8192):
    if not 64 <= max_nodes <= 30000:
        raise ValueError("SHAP component node limit must be between 64 and 30000")
    components = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        if tree.max_depth > 24:
            raise ValueError("SHAP adapter supports the approved depth-10/depth-20 forests")
        sizes = np.ones(tree.node_count, dtype=np.int64)
        for node in range(tree.node_count - 1, -1, -1):
            if tree.children_left[node] >= 0:
                sizes[node] += sizes[tree.children_left[node]] + sizes[tree.children_right[node]]
        thresholds = tree.threshold.astype(np.float32)
        thresholds = np.where(thresholds.astype(np.float64) > tree.threshold,
                              np.nextafter(thresholds, np.float32(-np.inf)), thresholds)
        probabilities = tree.value[:, 0, :].astype(np.float64)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        probabilities /= len(model.estimators_)

        def component(root, path):
            left, right, features, cuts, values = [], [], [], [], []

            def allocate(original=None):
                index = len(left)
                left.append(-1); right.append(-1)
                features.append(-2 if original is None else int(tree.feature[original]))
                cuts.append(-2.0 if original is None else float(thresholds[original]))
                values.append(np.zeros(2) if original is None else probabilities[original])
                return index

            def copy_subtree(node):
                index = allocate(node)
                if tree.children_left[node] >= 0:
                    left[index] = copy_subtree(tree.children_left[node])
                    right[index] = copy_subtree(tree.children_right[node])
                return index

            def gated(depth):
                if depth == len(path):
                    return copy_subtree(root)
                node, goes_left = path[depth]
                index = allocate(node)
                if goes_left:
                    left[index] = gated(depth + 1)
                    right[index] = allocate()
                else:
                    left[index] = allocate()
                    right[index] = gated(depth + 1)
                return index

            gated(0)
            if len(left) > max_nodes:
                raise ValueError("SHAP component exceeded its node limit")
            components.append({
                "children_left": np.asarray(left), "children_right": np.asarray(right),
                "children_default": np.asarray(left), "features": np.asarray(features),
                "thresholds": np.asarray(cuts, dtype=np.float64),
                "values": np.asarray(values, dtype=np.float64),
                "node_sample_weight": np.ones(len(left), dtype=np.float64),
            })

        def partition(node, path):
            if sizes[node] + 2 * len(path) <= max_nodes:
                component(node, path)
            else:
                partition(tree.children_left[node], path + [(node, True)])
                partition(tree.children_right[node], path + [(node, False)])

        partition(0, [])
    return {"trees": components, "tree_output": "probability",
            "objective": "binary_crossentropy", "input_dtype": np.float32,
            "internal_dtype": np.float64, "base_offset": 0.0}
