"""Synthetic research/selection/SHAP integration with held-out test guards."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from integritree.config import load_experiment
from integritree.settings import BACKEND_ROOT
from integritree.ml.data import RAW_COLUMNS, file_sha256
from integritree.ml.preparation import prepare_dataset, load_prepared_split
from integritree.ml.training import train_models
from integritree.ml.research import evaluate_run, load_report
from integritree.ml.features import FEATURE_COLUMNS


@pytest.fixture
def experiment(tmp_path,raw_record):
    frame=pd.DataFrame([raw_record | {"step":i+1,"amount":float(i+1),"nameOrig":f"C_{i}",
              "type":"TRANSFER" if i%2 else "CASH_OUT","isFraud":int(i%5==0)}
              for i in range(100)]).loc[:,RAW_COLUMNS]
    source=tmp_path / "synthetic.csv"; frame.to_csv(source,index=False)
    config=load_experiment(BACKEND_ROOT / "configs/experiment.yaml").model_copy(deep=True)
    config.dataset=config.dataset.model_copy(update={"filename":source.name,"sha256":file_sha256(source),
                                                      "row_count":len(frame)})
    prepared=prepare_dataset(source,config,tmp_path / "prepared","tiny",progress=lambda _:None)
    baseline=train_models(prepared,config,tmp_path / "models","baseline",progress=lambda _:None)
    return prepared,baseline,config


def test_baseline_validation_reports_and_test_guard(experiment,tmp_path,monkeypatch):
    prepared,baseline,config=experiment
    output=evaluate_run(baseline,prepared,config,tmp_path / "reports","validation","validate",lambda _:None)
    meta,preds=load_report(output)
    assert meta["split"]=="validation" and not meta["official_test"]
    assert len(preds)==10 and preds.transaction_id.is_unique
    assert preds["threshold"].eq(.5).all()
    assert config.evaluation.pr_auc_method=="average_precision"
    csv=pd.read_csv(output / "predictions.csv")
    np.testing.assert_allclose(csv.rf_risk_score,preds.rf_risk_score,rtol=0,atol=1e-15)
    def forbidden(*args,**kwargs):
        raise AssertionError("Test data must not be loaded")
    monkeypatch.setattr("integritree.ml.research.load_prepared_split",forbidden)
    with pytest.raises(ValueError,match="frozen"):
        evaluate_run(baseline,prepared,config,tmp_path / "reports","test")
    with (output / "metrics.json").open("a") as handle: handle.write(" ")
    with pytest.raises(ValueError,match="fingerprint"):
        load_report(output)


def test_real_tree_shap_cache_reconstruction_and_coverage(experiment,tmp_path):
    from integritree.ml.explainability import ExplanationEngine, explain_report, summarize
    prepared,baseline,config=experiment
    X,y,ids=load_prepared_split(prepared,"validation")
    engine=ExplanationEngine(baseline,prepared,config.shap,tmp_path / "cache")
    before=engine.background.copy()
    result=engine.explain(X.iloc[:1],["synthetic-1"])
    assert len(result)==2
    for item in result:
        assert abs(item["base_value"]+sum(f["contribution"] for f in item["features"])-item["output_value"])<=1e-6
        assert Path(item["waterfall_path"]).exists()
        assert len(item["features"])==11
        assert item["status"]=="computed"
    assert engine.explain(X.iloc[:1],["synthetic-1"])==result
    changed=X.iloc[:1].copy(); changed.iloc[0,0]+=.1
    fresh=engine.explain(changed,["synthetic-1"])
    assert fresh[0]["cache_key"]!=result[0]["cache_key"]
    np.testing.assert_array_equal(engine.background,before)
    top,_=summarize(np.zeros(11),["example"]*11,1e-9)
    assert top["status"]=="no_positive_contributor"
    top,text=summarize(np.array([.2,.2,-.4]+[0]*8),["example"]*11,1e-9)
    assert top["feature"]==FEATURE_COLUMNS[0]
    assert "decreasing" in text
    report=evaluate_run(baseline,prepared,config,tmp_path / "reports",progress=lambda _:None)
    explanation=explain_report(baseline,prepared,config,report,tmp_path / "explanations",
                               tmp_path / "cache","preview",1,progress=lambda _:None)
    metadata=json.loads((explanation / "metadata.json").read_text())
    assert metadata["rows_completed"]==1 and metadata["population_rows"]==10
    assert metadata["scope"]=="preview"
