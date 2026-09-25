"""Independent coalition oracle checks every adapted SHAP contribution."""
import math
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def test_gated_tree_sum_matches_model_and_exact_interventional_values(monkeypatch, tmp_path):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    import shap
    from integritree.ml.shap_adapter import forest_representation
    rng = np.random.default_rng(42)
    X = rng.random((500, 3), dtype=np.float32)
    y = rng.integers(0, 2, 500)
    model = RandomForestClassifier(n_estimators=3, max_depth=10, random_state=42).fit(X, y)
    original = [tree.tree_.threshold.copy() for tree in model.estimators_]
    representation = forest_representation(model, max_nodes=64)
    assert len(representation["trees"]) > 3
    assert max(len(tree["features"]) for tree in representation["trees"]) <= 64
    background = X[:8]
    engine = shap.TreeExplainer(representation, background, model_output="probability",
                                feature_perturbation="interventional")
    np.testing.assert_allclose(engine.model.predict(X), model.predict_proba(X), atol=1e-12, rtol=0)
    for x in X[15:18]:
        game = {}
        for mask in range(8):
            hybrid = background.copy()
            for j in range(3):
                if mask & (1 << j):
                    hybrid[:, j] = x[j]
            game[mask] = model.predict_proba(hybrid)[:, 1].mean()
        exact = np.zeros(3)
        for j in range(3):
            for mask in range(8):
                if not mask & (1 << j):
                    size = mask.bit_count()
                    weight = math.factorial(size) * math.factorial(2-size) / math.factorial(3)
                    exact[j] += weight * (game[mask | (1 << j)] - game[mask])
        result = engine(x[None, :], check_additivity=False)
        np.testing.assert_allclose(result.values[0, :, 1], exact, atol=1e-7, rtol=0)
        assert abs(result.base_values[0, 1] - game[0]) < 1e-12
    for before, tree in zip(original, model.estimators_):
        np.testing.assert_array_equal(before, tree.tree_.threshold)
