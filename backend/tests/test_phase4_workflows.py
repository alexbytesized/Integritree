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
from integritree.ml.artifacts import load_bundle
from integritree.ml.research import evaluate_run, load_report
from integritree.ml.selection import select_models, choose_configuration, choose_threshold
from integritree.ml.inference import predict_records, predict_features
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


def test_tie_breaking_without_float_distance_bias():
    rows=[{"candidate":list(c),"mean_f1_fraction":"1/2"} for c in [(100,10),(100,20),(200,10),(200,20)]]
    assert choose_configuration(rows)["candidate"]==[100,10]
    threshold,_=choose_threshold([0,1],[0,1],[0,1])
    assert threshold==.5
    # Maxima at 0.45 and 0.55 equidistant logic is separately represented by integer steps.
    with pytest.raises(ValueError):
        choose_configuration(rows[:3])
    with pytest.raises(ValueError,match="both"):
        choose_threshold([0,0],[0,0],[0,0])


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


def test_selection_preserves_baseline_and_only_then_allows_test(experiment,tmp_path,monkeypatch):
    prepared,baseline,config=experiment
    baseline_hash=file_sha256(baseline / "metadata.json")
    import integritree.ml.research as research
    real=research.load_prepared_split
    calls=[]
    def tracked(path,split):
        calls.append(split)
        assert split=="validation"
        return real(path,split)
    monkeypatch.setattr(research,"load_prepared_split",tracked)
    selected=select_models(baseline,prepared,config,tmp_path / "selections","selected",progress=lambda _:None)
    assert calls==["validation"]*4
    bundle=load_bundle(selected)
    assert bundle.metadata["stage"]=="validation_selected"
    assert bundle.config.random_forest.tuning_procedure=="validation_grid_selected"
    assert file_sha256(baseline / "metadata.json")==baseline_hash
    assert load_bundle(baseline).config.scoring.threshold==.5
    monkeypatch.setattr(research,"load_prepared_split",real)
    result=evaluate_run(selected,prepared,config,tmp_path / "reports","test",progress=lambda _:None)
    meta,_=load_report(result)
    assert meta["official_test"] is True  # Synthetic test fixture only.
    effective=json.loads((result / "evaluation_configuration.json").read_text())
    assert effective["effective_scoring"]["threshold"]==bundle.config.scoring.threshold
    assert effective["effective_random_forest"]["tuning_procedure"]=="validation_grid_selected"
    with (selected / "selection.json").open("a") as handle: handle.write(" ")
    with pytest.raises(ValueError,match="fingerprint"):
        load_bundle(selected)


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


def test_interrupted_selection_resumes_without_retraining_completed_candidates(experiment,tmp_path,monkeypatch):
    import integritree.ml.training as training
    prepared,baseline,config=experiment
    original=training.train_models
    calls=[]
    def interrupted(*args,**kwargs):
        calls.append((args[1].random_forest.n_estimators,args[1].random_forest.max_depth))
        if len(calls)==2:
            raise MemoryError("Simulated resource interruption")
        return original(*args,**kwargs)
    monkeypatch.setattr(training,"train_models",interrupted)
    with pytest.raises(MemoryError):
        select_models(baseline,prepared,config,tmp_path / "selections","resume",progress=lambda _:None)
    partial=tmp_path / "selections/resume"
    completed=json.loads((partial / "search_progress.json").read_text())["completed_candidates"]
    assert len(completed)==2
    def resumed(*args,**kwargs):
        calls.append((args[1].random_forest.n_estimators,args[1].random_forest.max_depth))
        return original(*args,**kwargs)
    monkeypatch.setattr(training,"train_models",resumed)
    result=select_models(baseline,prepared,config,tmp_path / "selections",resume=partial,progress=lambda _:None)
    assert result==partial.resolve()
    assert calls==[(100,10),(200,10),(200,10),(200,20)]
    final=json.loads((result / "selection.json").read_text())
    assert final["candidates"][:2]==completed
    assert load_bundle(result).metadata["stage"]=="validation_selected"


def test_selection_releases_previous_report_before_next_training(experiment,tmp_path,monkeypatch):
    import weakref
    import integritree.ml.selection as selection
    import integritree.ml.training as training
    prepared,baseline,config=experiment
    real_load=selection.load_report
    real_train=training.train_models
    report_frames=[]
    observed_previous_reports=[]
    def tracked_load(path):
        metadata,frame=real_load(path)
        report_frames.append(weakref.ref(frame))
        return metadata,frame
    def checked_train(*args,**kwargs):
        observed_previous_reports.append(len(report_frames))
        assert all(reference() is None for reference in report_frames), "Previous prediction table is still live during training"
        return real_train(*args,**kwargs)
    monkeypatch.setattr(selection,"load_report",tracked_load)
    monkeypatch.setattr(training,"train_models",checked_train)
    selected=select_models(baseline,prepared,config,tmp_path / "selections","memory_cleanup",progress=lambda _:None)
    assert observed_previous_reports==[0,2,3]
    assert load_bundle(selected).metadata["stage"]=="validation_selected"
